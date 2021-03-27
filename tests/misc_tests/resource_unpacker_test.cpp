#include <tests/common/helpers.hpp>
#include <fstream>
#include <misc/resource_unpacker.hpp>

namespace misc
{
    bool operator==(const IdxFileFormat& a, const IdxFileFormat& b)
    {
        return a.magic_number == b.magic_number && a.dimensions == b.dimensions && a.data == b.data;
    }
}

namespace misc::resource_unpacker
{
    using misc::IdxFileFormat;

    const std::string PATH_PREF = "../../tests/misc_tests/";

    template<typename T>
    void write_bytes(std::ofstream& os, T data)
    {
        os.write((char*)&data, sizeof(data));
    }

    void create_idx_file(const std::string& filename, const IdxFileFormat& idx_file)
    {
        std::ofstream outfile(filename, std::ios::out | std::ios::trunc | std::ios::binary);
        assert(outfile);

        write_bytes(outfile, idx_file.magic_number);
        for (const auto& dim: idx_file.dimensions)
            write_bytes(outfile, dim);
        for (const auto& item: idx_file.data)
            write_bytes(outfile, item);

        outfile.close();
    }


    TEST(MiscResourceUnpackerTest, OneDimensional)
    {
        const std::string filename = PATH_PREF + "ru_test_1.idx1-ubyte";
        const std::vector<uint8_t> data{ 67U,65U,78U,65U,82U,89U };
        const IdxFileFormat idx_file{
            .magic_number = 0x00000801U,
            .dimensions = { static_cast<uint32_t>(data.size()) },
            .data = data
        };

        //create_idx_file(filename, idx_file);

        EXPECT_EQ(IdxFileUnpacker::unpack(filename), idx_file);
    }

    TEST(MiscResourceUnpackerTest, TwoDimensional)
    {
        const std::string filename = PATH_PREF + "ru_test_2.idx2-ubyte";
        const std::vector<uint8_t> data{ 67U,65U,78U,65U,82U,84U };
        const IdxFileFormat idx_file{
            .magic_number = 0x00000802U,
            .dimensions = { 3u, 2u },
            .data = data
        };

        ///create_idx_file(filename, idx_file);

        EXPECT_EQ(IdxFileUnpacker::unpack(filename), idx_file);
    }

    TEST(MiscResourceUnpackerTest, ThreeDimensional)
    {
        const std::string filename = PATH_PREF + "ru_test_3.idx3-ubyte";
        const std::vector<uint8_t> data{
            65u,66u,67u,68u,69u,70u,
            75u,76u,77u,78u,79u,80u,
            85u,86u,87u,88u,89u,90u,
            95u,96u,97u,98u,99u,100u,
            105u,106u,107u,108u,109u,110u,
        };
        const IdxFileFormat idx_file{
            .magic_number = 0x00000803U,
            .dimensions = { 5u, 2u, 3u },
            .data = data
        };

        //create_idx_file(filename, idx_file);

        EXPECT_EQ(IdxFileUnpacker::unpack(filename), idx_file);
    }
}