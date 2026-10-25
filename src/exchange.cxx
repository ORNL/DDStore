#include "exchange.hpp"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <atomic>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef DDSTORE_HAVE_MPI
void MpiExchange::allgather(const void *in, void *out, size_t bytes)
{
    if (MPI_Allgather(in, (int)bytes, MPI_BYTE, out, (int)bytes, MPI_BYTE, this->comm) != MPI_SUCCESS)
        throw std::runtime_error("MPI_Allgather failed");
}
#endif

std::string next_exchange_ns()
{
    static std::atomic<long> n(0);
    return "s" + std::to_string(n++);
}

static std::string exchange_session()
{
    const char *s = getenv("DDSTORE_SESSION");
    if (s && s[0])
        return s;
    const char *job = getenv("SLURM_JOB_ID");
    const char *step = getenv("SLURM_STEP_ID");
    if (job && job[0])
        return std::string(job) + "_" + (step && step[0] ? step : "0");
    return "0";
}

static double timeout_s()
{
    const char *e = getenv("DDSTORE_HANDSHAKE_TIMEOUT_S");
    return e ? atof(e) : 300.0;
}

static double now_s()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Write n bytes to path via a temp file and rename(), so a reader polling
 * path sees either nothing or the whole file.                            */
static void write_atomic(const std::string &path, const void *data, size_t n)
{
    std::string tmp = path + ".tmp";
    int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("FileExchange: cannot create " + tmp + ": " + strerror(errno));
    const char *p = (const char *)data;
    size_t left = n;
    while (left > 0)
    {
        ssize_t w = write(fd, p, left);
        if (w < 0)
        {
            if (errno == EINTR)
                continue;
            int e = errno;
            close(fd);
            unlink(tmp.c_str());
            throw std::runtime_error("FileExchange: write " + tmp + " failed: " + strerror(e));
        }
        p += w;
        left -= (size_t)w;
    }
    if (close(fd) != 0 || rename(tmp.c_str(), path.c_str()) != 0)
    {
        int e = errno;
        unlink(tmp.c_str());
        throw std::runtime_error("FileExchange: cannot publish " + path + ": " + strerror(e));
    }
}

/* Read exactly n bytes of path into buf. False if path does not exist (yet)
 * or has another size.                                                    */
static bool read_exact(const std::string &path, void *buf, size_t n)
{
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return false;
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size != n)
    {
        close(fd);
        return false;
    }
    char *p = (char *)buf;
    size_t left = n;
    while (left > 0)
    {
        ssize_t r = read(fd, p, left);
        if (r < 0 && errno == EINTR)
            continue;
        if (r <= 0)
        {
            close(fd);
            return false;
        }
        p += r;
        left -= (size_t)r;
    }
    close(fd);
    return true;
}

/* Poll until ready() returns true: 1 ms backoff doubling to 200 ms. */
template <typename F>
static void poll_until(F ready, const std::string &what)
{
    const double t0 = now_s(), limit = timeout_s();
    useconds_t wait = 1000;
    while (!ready())
    {
        if (now_s() - t0 > limit)
            throw std::runtime_error(
                "FileExchange: timed out after " + std::to_string((int)limit) +
                " s waiting for " + what +
                " (DDSTORE_HANDSHAKE_TIMEOUT_S; a rank may have failed, or stale files "
                "from an earlier run with the same DDSTORE_SESSION are in the way)");
        usleep(wait);
        wait = wait * 2 > 200000 ? 200000 : wait * 2;
    }
}

FileExchange::FileExchange(const std::string &dir, int rank, int size, const std::string &ns)
    : Exchange(rank, size), dir(dir), prefix(".x." + exchange_session() + "." + ns)
{
    if (size <= 0 || rank < 0 || rank >= size)
        throw std::invalid_argument("FileExchange: need 0 <= rank < size (got rank=" +
                                    std::to_string(rank) + ", size=" + std::to_string(size) + ")");
    mkdir(dir.c_str(), 0755); /* best effort; EEXIST is fine */
}

FileExchange::~FileExchange() {}

std::string FileExchange::path(long call, const char *who) const
{
    return this->dir + "/" + this->prefix + "." + std::to_string(call) + "." + who;
}

void FileExchange::allgather(const void *in, void *out, size_t bytes)
{
    const long k = this->ncalls++;
    const size_t head = 2 * sizeof(uint64_t); /* generation, nonce */
    const size_t slot = head + bytes;

    static thread_local std::mt19937_64 rng(std::random_device{}() ^ (uint64_t)getpid());
    const uint64_t nonce = rng();
    std::vector<char> mine(slot);
    memcpy(mine.data() + sizeof(uint64_t), &nonce, sizeof(nonce));
    memcpy(mine.data() + head, in, bytes);

    std::vector<char> all(slot * this->size);
    const std::string all_path = this->path(k, "all");
    const std::string go_path = this->path(0, "go");

    if (this->rank == 0)
    {
        /* A fresh generation for this exchange, before reading any rank's
         * file: files an earlier run left behind carry another one.        */
        if (k == 0)
        {
            do
                this->gen = rng();
            while (this->gen == 0);
            write_atomic(go_path, &this->gen, sizeof(this->gen));
        }
        memcpy(mine.data(), &this->gen, sizeof(this->gen));
        memcpy(all.data(), mine.data(), slot);
        for (int r = 1; r < this->size; r++)
        {
            const std::string p = this->path(k, std::to_string(r).c_str());
            char *dst = all.data() + r * slot;
            poll_until([&] {
                return read_exact(p, dst, slot) && memcmp(dst, &this->gen, sizeof(this->gen)) == 0;
            }, p + " (rank " + std::to_string(r) + ")");
        }
        write_atomic(all_path, all.data(), all.size());
        for (int r = 1; r < this->size; r++)
            unlink(this->path(k, std::to_string(r).c_str()).c_str());
        /* Every rank has read call k-1's file (and the generation): it
         * wrote its call-k file after doing so.                            */
        if (k > 0)
            unlink(this->path(k - 1, "all").c_str());
        if (k == 1)
            unlink(go_path.c_str());
    }
    else
    {
        const std::string my_path = this->path(k, std::to_string(this->rank).c_str());
        /* Call 0: take the generation from rank 0's go file. One left by an
         * earlier run may be read first; rank 0 then ignores this rank's
         * file, so re-read the go file while waiting and resend on change. */
        auto read_gen = [&] {
            uint64_t g;
            if (read_exact(go_path, &g, sizeof(g)) && g != 0 && g != this->gen)
            {
                this->gen = g;
                memcpy(mine.data(), &g, sizeof(g));
                write_atomic(my_path, mine.data(), slot);
            }
        };
        if (k == 0)
            poll_until([&] {
                read_gen();
                return this->gen != 0;
            }, go_path);
        else
        {
            memcpy(mine.data(), &this->gen, sizeof(this->gen));
            write_atomic(my_path, mine.data(), slot);
        }
        poll_until([&] {
            if (read_exact(all_path, all.data(), all.size()) &&
                memcmp(all.data() + this->rank * slot, mine.data(), head) == 0)
                return true;
            if (k == 0)
                read_gen();
            return false;
        }, all_path);
    }

    for (int r = 0; r < this->size; r++)
        memcpy((char *)out + r * bytes, all.data() + r * slot + head, bytes);
}
