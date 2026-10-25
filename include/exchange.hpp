#pragma once

#include <stddef.h>
#include <stdint.h>
#include <string>
#ifdef DDSTORE_HAVE_MPI
#include <mpi.h>
#endif

/* -----------------------------------------------------------------------
 * Exchange: the one collective DDStore's libfabric methods (1 and 2) need
 * when a variable is added -- every rank of the store gathering every
 * rank's handshake record. MpiExchange does it with MPI_Allgather;
 * FileExchange through files in a shared directory, so a store can be
 * built without MPI. Every rank must call allgather() the same number of
 * times, in the same order (add()/init() are collective).
 * ----------------------------------------------------------------------- */
class Exchange
{
public:
    Exchange(int rank, int size) : rank(rank), size(size) {}
    virtual ~Exchange() {}

    /* out (size * bytes) receives every rank's `bytes` from in, by rank.
     * Throws std::runtime_error on failure or timeout.                    */
    virtual void allgather(const void *in, void *out, size_t bytes) = 0;

    const int rank, size;
};

#ifdef DDSTORE_HAVE_MPI
class MpiExchange : public Exchange
{
public:
    MpiExchange(MPI_Comm comm, int rank, int size) : Exchange(rank, size), comm(comm) {}
    void allgather(const void *in, void *out, size_t bytes) override;

private:
    MPI_Comm comm;
};
#endif

/* File-based allgather in `dir` (on a filesystem all ranks see). Call k:
 *   1. every rank writes {dir}/.x.{session}.{ns}.{k}.{rank}
 *      (temp file + rename, so readers never see a partial file);
 *   2. rank 0 waits for all `size` files, writes the combined
 *      {dir}/.x.{session}.{ns}.{k}.all the same way, and removes the
 *      per-rank files and the previous call's combined file;
 *   3. every other rank waits for the combined file and reads it.
 * Only rank 0 polls for `size` files; the others poll for one, with a
 * backoff from 1 ms to 200 ms (DDSTORE_HANDSHAKE_TIMEOUT_S, default 300).
 *
 * session: DDSTORE_SESSION, else SLURM_JOB_ID.SLURM_STEP_ID, else "0".
 * ns separates stores (and the Python layer's collectives) of one session;
 * all ranks must create them in the same order with the same ns.
 * Files left by an earlier run with the same session and ns are never
 * mistaken for this one's:
 *   - on call 0, rank 0 picks a random generation and writes it to
 *     {dir}/.x.{session}.{ns}.0.go first; every rank's file carries it,
 *     and rank 0 accepts only files with it. A rank that read an earlier
 *     run's go file re-reads it while waiting and resends its file;
 *   - each rank's contribution also carries a random nonce, which the
 *     rank checks in the combined file.
 * Rank 0 removes the go file on call 1.                                   */
class FileExchange : public Exchange
{
public:
    FileExchange(const std::string &dir, int rank, int size, const std::string &ns);
    ~FileExchange();
    void allgather(const void *in, void *out, size_t bytes) override;

private:
    std::string path(long call, const char *who) const;

    std::string dir, prefix; /* prefix: ".x.{session}.{ns}" */
    long ncalls = 0;
    uint64_t gen = 0; /* this exchange's generation, from rank 0 on call 0 */
};

/* Default namespace for a store's FileExchange: "s0", "s1", ... in
 * construction order within this process.                                 */
std::string next_exchange_ns();
