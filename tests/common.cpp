#include "common.h"

extern "C" {
    INCTXT(testsStructsHeaderBin, DAPPA_SOURCE_DIR "/tests/structs.h");
}

bool printTestHeader(std::ofstream &file, const std::string &stage)
{
    file << stage << std::endl << "----------------" << std::endl;
    return true;
}

bool printTestHeader(std::ofstream &file, const std::string &stage, uint64_t actual, uint64_t target)
{
    file << stage << " (" << actual << " elements)";
    if (actual != target)
        file << " [Wrong, should be " << target << "]";
    file << std::endl << "----------------" << std::endl;
    return actual == target;
}
