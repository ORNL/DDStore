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
    std::vector<int> lenlist; // list of sizes of all element
    std::vector<long unsigned int> dataoffsets; // list of offsets of all elements
    std::vector<int> offsets; // list of offsets at each process
    // std::vector<int> offsetlist; // list of offsets at each process
    // std::vector<int> counts;  // list of total number of elements each process has
    // std::vector<int> process_offsets; // list of offsets at each process
    MPI_Win win;
    bool active;
    bool fence_active;
};
typedef struct VarInfo VarInfo_t;

int sortedsearch(std::vector<int> &vec, long unsigned int idx);

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

    /*
    ** Data layout sketch:
    ** We assume an array of variable-length data per process.
    ** Each element in the array is a block of n x disp elements (n: len, disp:
    *displacement)
    ** E.g.,
    **      +-------+-------+-------+
    **      |<- 1 ->|<- 2 ->|<- 3 ->|
    **      +-------+-------+-------+
    ** len is width (variable)
    ** disp is height (same for all)
    */

    template <typename T>
    void create(std::string name, T *buffer, int disp, int *l_lenlist, int ncount)
    {
        long l_ntotal = 0;
        for (int i = 0; i < ncount; i++)
            l_ntotal += l_lenlist[i];

        MPI_Win win;
        MPI_Win_create(buffer,                              /* pre-allocated buffer */
                       (MPI_Aint)l_ntotal * disp * sizeof(T), /* size in bytes */
                       disp * sizeof(T),                    /* displacement units */
                       MPI_INFO_NULL,                       /* info object */
                       this->comm,                          /* communicator */
                       &win);                               /* window object */

        // Use MPI_Allgatherv
        // Note: expect possible limit for using int32 (due to MPI)
        std::vector<int> recvcounts(this->comm_size);
        std::vector<int> displs(this->comm_size);

        MPI_Allgather(&ncount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, this->comm);
        int ntotal = 0;
        {
            long unsigned int i = 0;
            for (auto &x : recvcounts)
            {
                displs[i++] = ntotal;
                ntotal += x;
            }
        }

        // for (long unsigned int i = 0; i < recvcounts.size(); i++)
        //     std::cout << this->rank << ": " << "recvcounts[" << i << "] = " << recvcounts[i] << std::endl;
        // for (long unsigned int i = 0; i < displs.size(); i++)
        //     std::cout << this->rank << ": " << "displs[" << i << "] = " << displs[i] << std::endl;
        // std::cout << this->rank << ": " << "ntotal = " << ntotal << std::endl;

        std::vector<int> lenlist(ntotal);
        std::vector<long unsigned int> dataoffsets(ntotal);
        /*
        int MPI_Allgatherv(const void *sendbuf, int sendcount, MPI_Datatype
                            sendtype, void *recvbuf, const int *recvcounts, const int *displs,
                            MPI_Datatype recvtype, MPI_Comm comm)
        */
        MPI_Allgatherv(l_lenlist, ncount, MPI_INT, lenlist.data(), recvcounts.data(), displs.data(), MPI_INT, this->comm);
        {
            long unsigned int sum = 0;
            long unsigned int i = 0;
            for (auto &x : lenlist)
            {
                dataoffsets[i++] = sum;
                sum += x;
            }
        }

        // for (long unsigned int i = 0; i < lenlist.size(); i++)
        //     std::cout << this->rank << ": " << "lenlist[" << i << "] = " << lenlist[i] << std::endl;
        // for (long unsigned int i = 0; i < dataoffsets.size(); i++)
        //     std::cout << this->rank << ": " << "dataoffsets[" << i << "] = " << dataoffsets[i] << std::endl;

        int max_disp = 0;
        // We assume disp is same for all
        MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
        if (max_disp != disp)
            throw std::invalid_argument("Invalid disp");

        VarInfo_t var;
        var.name = name;
        var.typeinfo = typeid(T).name();
        var.disp = disp;
        var.win = win;
        var.lenlist = lenlist;
        var.dataoffsets = dataoffsets;
        var.offsets = displs;
        var.active = true;
        var.fence_active = false;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    template <typename T>
    void get(std::string name, long unsigned int id, T *buffer)
    {
        VarInfo_t varinfo = this->varlist[name];

        if (varinfo.typeinfo != typeid(T).name())
            throw std::invalid_argument("Invalid data type");

        int target = sortedsearch(varinfo.offsets, id);
        int offset = varinfo.dataoffsets[varinfo.offsets[target]];
        int len = varinfo.lenlist[id];
        long unsigned int dataoffset = varinfo.dataoffsets[id];
        // std::cout << "[" << this->rank << "] id,target,offset,dataoffset,len: " << id << "," << target << "," << offset << "," << dataoffset << "," << len << std::endl;

        MPI_Win win = varinfo.win;
        MPI_Win_lock(MPI_LOCK_SHARED, target, 0, win);
        /*
        int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype
                    origin_datatype, int target_rank, MPI_Aint target_disp,
                    int target_count, MPI_Datatype target_datatype, MPI_Win
                    win)
        */
        MPI_Get(buffer,                         /* pre-allocated buffer on RMA origin process */
                varinfo.disp * sizeof(T) * len, /* count on RMA origin process */
                MPI_BYTE,                       /* type on RMA origin process */
                target,                         /* rank of RMA target process */
                dataoffset - offset,            /* displacement on RMA target process */
                varinfo.disp * sizeof(T) * len, /* count on RMA target process */
                MPI_BYTE,                       /* type on RMA target process */
                win);                           /* window object */
        MPI_Win_unlock(target, win);
    }

  private:
    MPI_Comm comm;
    int comm_size;
    int rank;

    std::map<std::string, VarInfo_t> varlist;
};
