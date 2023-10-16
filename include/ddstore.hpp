#include <iostream>
#include <map>
#include <mpi.h>
#include <string>
#include <typeinfo>
#include <vector>

struct VarInfo
{
    std::string name;
    std::string typeinfo;
    int disp;
    std::vector<long> lenlist;
    MPI_Win win;
    bool active;
    bool fence_active;
    void *base;
};
typedef struct VarInfo VarInfo_t;

int sortedsearch(std::vector<long> &vec, long num);

class DDStore
{
  public:
    DDStore();
    DDStore(MPI_Comm comm);
    ~DDStore();

    void query(std::string name, VarInfo_t &varinfo);
    void epoch_begin();
    void epoch_end();
    void free();

    template <typename T> void add(std::string name, T *buffer, long nrows, int disp)
    {
        void *base;
        int err = MPI_Alloc_mem((MPI_Aint)(nrows * disp * sizeof(T)), MPI_INFO_NULL, &base);
        if (err)
        {
            exit(1);
        }
        memcpy(base, buffer, nrows * disp * sizeof(T));

        MPI_Win win;
        MPI_Win_create(base,                             /* pre-allocated buffer */
                       (MPI_Aint)nrows * disp * sizeof(T), /* size in bytes */
                       disp * sizeof(T),                   /* displacement units */
                       MPI_INFO_NULL,                      /* info object */
                       this->comm,                         /* communicator */
                       &win /* window object */);

        std::vector<long> lenlist(this->comm_size);
        MPI_Allgather(&nrows, 1, MPI_LONG, lenlist.data(), 1, MPI_LONG, this->comm);

        int max_disp = 0;
        // We assume disp is same for all
        MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
        if (max_disp != disp)
            throw std::invalid_argument("Invalid disp");

        long sum = 0;
        for (long unsigned int i = 0; i < lenlist.size(); i++)
        {
            sum += lenlist[i];
            lenlist[i] = sum;
        }
        // for (long unsigned int i = 0; i < lenlist.size(); i++)
        // {
        //     std::cout << "lenlist[" << i << "]: " << lenlist[i] << std::endl;
        // }
        // std::cout << "sum: " << sum << std::endl;

        VarInfo_t var;
        var.name = name;
        var.typeinfo = typeid(T).name();
        var.disp = disp;
        var.win = win;
        var.lenlist = lenlist;
        var.active = true;
        var.fence_active = false;
        var.base = base;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    template <typename T> void get(std::string name, long start, long count, T *buffer)
    {
        VarInfo_t varinfo = this->varlist[name];

        if (varinfo.typeinfo != typeid(T).name())
            throw std::invalid_argument("Invalid data type");

        int target = sortedsearch(varinfo.lenlist, start);
        long offset = target > 0 ? varinfo.lenlist[target - 1] : 0;
        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        if (start < offset)
            throw std::invalid_argument("Invalid start on target");

        if ((start + count) > varinfo.lenlist[target])
            throw std::invalid_argument("Invalid count on target");

        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        MPI_Win win = varinfo.win;
        MPI_Win_lock(MPI_LOCK_SHARED, target, 0, win);
        /*
        int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
                    origin_datatype, int target_rank, MPI_Aint target_disp,
                    int target_count, MPI_Datatype target_datatype, MPI_Win
                    win)
        */
        MPI_Get(buffer,                           /* pre-allocated buffer on RMA origin process */
                varinfo.disp * sizeof(T) * count, /* count on RMA origin process */
                MPI_BYTE,                         /* type on RMA origin process */
                target,                           /* rank of RMA target process */
                start - offset,                   /* displacement on RMA target process */
                varinfo.disp * sizeof(T) * count, /* count on RMA target process */
                MPI_BYTE,                         /* type on RMA target process */
                win /* window object */);
        MPI_Win_unlock(target, win);
    }

  private:
    MPI_Comm comm;
    int comm_size;
    int rank;

    std::map<std::string, VarInfo_t> varlist;
};
