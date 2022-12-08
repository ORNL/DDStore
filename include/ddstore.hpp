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
    int ngroup;
    int gid;
    int gsize;
    std::vector<int> lenlist;
    MPI_Win win;
    bool active;
    bool fence_active;
};
typedef struct VarInfo VarInfo_t;

int sortedsearch(std::vector<int> &vec, int num);

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

    template <typename T> void add(std::string name, T *buffer, int nrows, int disp, int gid = 0)
    {
        MPI_Win win;
        MPI_Win_create(buffer,                             /* pre-allocated buffer */
                       (MPI_Aint)nrows * disp * sizeof(T), /* size in bytes */
                       disp * sizeof(T),                   /* displacement units */
                       MPI_INFO_NULL,                      /* info object */
                       this->comm,                         /* communicator */
                       &win /* window object */);

        std::vector<int> lenlist(this->comm_size);
        MPI_Allgather(&nrows, 1, MPI_INT, lenlist.data(), 1, MPI_INT, this->comm);

        int max_disp = 0;
        // We assume disp is same for all
        MPI_Allreduce(&disp, &max_disp, 1, MPI_INT, MPI_MAX, this->comm);
        if (max_disp != disp)
            throw std::invalid_argument("Invalid disp");

        std::vector<int> gidlist(this->comm_size);
        MPI_Allgather(&gid, 1, MPI_INT, gidlist.data(), 1, MPI_INT, this->comm);
        int maxgid = 0;
        for (long unsigned int i = 0; i < gidlist.size(); i++)
        {
            if (gidlist[i] > maxgid)
                maxgid = gidlist[i];
        }

        int ngroup = maxgid + 1;
        int gsize = this->comm_size / ngroup;
        // std::cout << "ngroup: " << ngroup << std::endl;
        // std::cout << "gsize: " << gsize << std::endl;
        if (this->comm_size % gsize > 0)
            throw std::invalid_argument("Invalid group size");

        for (int i = 0; i < gsize; i++)
        {
            for (int j = 1; j <= maxgid; j++)
            {
                if (lenlist[i] != lenlist[i + j * gsize])
                {
                    throw std::invalid_argument("Invalid len");
                }
            }
        }

        int sum = 0;
        for (long unsigned int i = 0; i < lenlist.size(); i++)
        {
            sum += lenlist[i];
            lenlist[i] = sum;
        }
        lenlist.resize(gsize);
        // for (long unsigned int i = 0; i < lenlist.size(); i++)
        // {
        //     std::cout << "lenlist[" << i << "]: " << lenlist[i] << std::endl;
        // }
        // std::cout << "sum: " << sum << std::endl;

        VarInfo_t var;
        var.name = name;
        var.typeinfo = typeid(T).name();
        var.disp = disp;
        var.ngroup = ngroup;
        var.gid = gid;
        var.gsize = gsize;
        var.win = win;
        var.lenlist = lenlist;
        var.active = true;
        var.fence_active = false;

        this->varlist.insert(std::pair<std::string, VarInfo_t>(name, var));
    }

    template <typename T> void get(std::string name, int start, int count, T *buffer)
    {
        VarInfo_t varinfo = this->varlist[name];

        if (varinfo.typeinfo != typeid(T).name())
            throw std::invalid_argument("Invalid data type");

        int target = sortedsearch(varinfo.lenlist, start);
        int offset = target > 0 ? varinfo.lenlist[target - 1] : 0;
        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        if (start < offset)
            throw std::invalid_argument("Invalid start on target");

        if ((start + count) > varinfo.lenlist[target])
            throw std::invalid_argument("Invalid count on target");

        int _target = target;
        if (varinfo.ngroup > 0)
            _target = target + varinfo.gid * varinfo.gsize;
        // std::cout << "target,offset,start,count: " << target << "," << offset << "," << start << "," << count <<
        // std::endl;

        MPI_Win win = varinfo.win;
        MPI_Win_lock(MPI_LOCK_SHARED, _target, 0, win);
        MPI_Get(buffer,                           /* pre-allocated buffer on RMA origin process */
                varinfo.disp * sizeof(T) * count, /* count on RMA origin process */
                MPI_BYTE,                         /* type on RMA origin process */
                _target,                          /* rank of RMA target process */
                start - offset,                   /* displacement on RMA target process */
                varinfo.disp * sizeof(T) * count, /* count on RMA target process */
                MPI_BYTE,                         /* type on RMA target process */
                win /* window object */);
        MPI_Win_unlock(_target, win);
    }

  private:
    MPI_Comm comm;
    int comm_size;
    int rank;

    std::map<std::string, VarInfo_t> varlist;
};
