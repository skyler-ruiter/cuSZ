/**
 * @file datasets.cc
 * @author Jiannan Tian
 * @brief Demonstrative datasets with prefilled dimensions from https://sdrbench.github.io
 * @version 0.1
 * @date 2020-09-20
 * Created on 2020-02-11
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <string>
#include <unordered_map>

#include "datasets.hh"
#include "metadata.hh"

// TODO need new buffer for extremely large input dimension
size_t dims_HACC_1GB[]    = {280953867, 1, 1, 1, 1};
size_t dims_HACC_1b[]     = {1073726487, 1, 1, 1, 1};  // 1-billion particles
size_t dims_CESM[]        = {3600, 1800, 1, 1, 2};
size_t dims_Hurricane[]   = {500, 500, 100, 1, 3};
size_t dims_NYX_s[]       = {512, 512, 512, 1, 3};
size_t dims_NYX_m[]       = {1024, 1024, 1024, 1, 3};
size_t dims_QMCPACK1[]    = {288, 69, 7935, 1, 3};
size_t dims_QMCPACK2[]    = {69, 69, 33120, 1, 3};
size_t dims_EXAFEL_demo[] = {388, 59200, 1, 1, 2};
size_t dims_ARAMCO[]      = {235, 849, 849, 1, 3};
size_t dims_parihaka[]    = {1168, 1126, 922, 1, 3};

size_t* InitializeDemoDims(
    std::string const& datum,
    size_t             cap,
    bool               override,
    size_t             new_d0,
    size_t             new_d1,
    size_t             new_d2,
    size_t             new_d3)
{
    std::unordered_map<std::string, size_t*>  //
        dataset_entries = {
            {std::string("hacc"), dims_HACC_1GB},      {std::string("hacc1b"), dims_HACC_1b},
            {std::string("cesm"), dims_CESM},          {std::string("hurricane"), dims_Hurricane},
            {std::string("nyx-s"), dims_NYX_s},        {std::string("nyx-m"), dims_NYX_m},
            {std::string("qmc"), dims_QMCPACK1},       {std::string("qmcpre"), dims_QMCPACK2},
            {std::string("exafel"), dims_EXAFEL_demo}, {std::string("aramco"), dims_ARAMCO},
            {std::string("parihaka"), dims_parihaka}  //
        };

    auto dims_L16 = new size_t[16]();
    int  BLK;

    if (not override) {
        auto dims_datum = dataset_entries.at(datum);
        std::copy(dims_datum, dims_datum + 4, dims_L16);
        dims_L16[nDIM] = dims_datum[4];
    }
    else {
        size_t dims_override[] = {new_d0, new_d1, new_d2, new_d3};
        std::copy(dims_override, dims_override + 4, dims_L16);
        dims_L16[nDIM] = ((size_t)new_d0 != 1) + ((size_t)new_d1 != 1) + ((size_t)new_d2 != 1) + ((size_t)new_d3 != 1);
    }

    if (dims_L16[nDIM] == 1)
        BLK = MetadataTrait<1>::Block;
    else if (dims_L16[nDIM] == 2)
        BLK = MetadataTrait<2>::Block;
    else if (dims_L16[nDIM] == 3)
        BLK = MetadataTrait<3>::Block;

    dims_L16[nBLK0]  = (dims_L16[DIM0] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK1]  = (dims_L16[DIM1] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK2]  = (dims_L16[DIM2] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK3]  = (dims_L16[DIM3] - 1) / (size_t)BLK + 1;
    dims_L16[LEN]    = dims_L16[DIM0] * dims_L16[DIM1] * dims_L16[DIM2] * dims_L16[DIM3];
    dims_L16[CAP]    = cap;
    dims_L16[RADIUS] = cap / 2;

    return dims_L16;
}