#include <iostream>
#include <memory>
#include <vector>
#include <math.h>
#include <chrono>
#include <stdexcept>

#include <kompute/Kompute.hpp>

static std::vector<float> get_data(size_t size)
{
    std::vector<float> dat(size);

/*
    float q = 0;
    const float mod = 309.109293751;

    for( float &f : dat )
    {
        f = q;
        q += 1;
        if( q > mod ) q -= mod;
    }
*/
    return dat;
}

static std::vector<uint32_t> get_spirv(const std::string &path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if( size == -1 || ( size & (sizeof(uint32_t) - 1) ) != 0 )
        throw std::runtime_error("bad file " + path);

    std::vector<uint32_t> spv(size/sizeof(uint32_t));

    if( file.read((char*)spv.data(), size) )
        return spv;

    throw std::runtime_error("bad read " + path);
}

std::pair<double, std::vector<float>> time_gpu(size_t size, size_t tests)
{
    kp::Manager mgr(1);

    auto data = get_data(size);

    const kp::Constants cst(size);

    const auto input_type = kp::Tensor::TensorTypes::eHost;
    const auto output_type = kp::Tensor::TensorTypes::eHost;

    auto Open = mgr.tensor(cst, input_type);
    auto High = mgr.tensor(cst, input_type);
    auto Low = mgr.tensor(cst, input_type);
    auto Close = mgr.tensor(cst, input_type);
    auto Result = mgr.tensor(cst, output_type);

    std::vector<std::shared_ptr<kp::Tensor>> tmps;

    for( int i = 0; i < 7; ++i )
        tmps.push_back( mgr.tensor(cst, kp::Tensor::TensorTypes::eStorage) );

    auto gpu_sum = get_spirv("sum.comp.spv");
    auto gpu_div_const = get_spirv("div_const.comp.spv");
    auto gpu_MA_const = get_spirv("MA_const.comp.spv");
    auto gpu_sub = get_spirv("sub.comp.spv");

    std::vector<float> res(size);

    const kp::Workgroup workgroup{ static_cast<uint32_t>( (size + 63) / 64 ), 1, 1 };
    std::vector<uint32_t> size_push{{ static_cast<uint32_t>(size) }};

    auto alg_1 = mgr.algorithm({Open, High, tmps[6]}, gpu_sum, workgroup, {}, size_push);
    auto alg_2 = mgr.algorithm({tmps[6], Low, tmps[0]}, gpu_sum, workgroup, {}, size_push);
    auto alg_3 = mgr.algorithm({tmps[0], Close, tmps[1]}, gpu_sum, workgroup, {}, size_push);

    auto alg_div_4 = mgr.algorithm({tmps[1], tmps[2]}, gpu_div_const, workgroup, {4.f}, size_push);

    auto alg_ma_20 = mgr.algorithm({tmps[2], tmps[3]}, gpu_MA_const, workgroup, {20}, size_push );
    auto alg_ma_40 = mgr.algorithm({tmps[2], tmps[4]}, gpu_MA_const, workgroup, {40}, size_push );

    auto alg_sub = mgr.algorithm({tmps[3], tmps[4], tmps[5]}, gpu_sub, workgroup, {}, size_push );

    auto seq = mgr.sequence();

    auto start = std::chrono::high_resolution_clock::now();

    for( size_t i = 0; i < tests; ++i )
    {
        Open->setData(data);
        High->setData(data);
        Low->setData(data);
        Close->setData(data);

        // CPU --> GPU
        seq->record<kp::OpTensorSyncDevice>({Open, High, Low, Close});

        // t1 = ((Open + High) + Low) + Close
        seq->record<kp::OpAlgoDispatch>(alg_1)
            ->record<kp::OpMemoryBarrier>({tmps[6]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader )
            ->record<kp::OpAlgoDispatch>(alg_2)
            ->record<kp::OpMemoryBarrier>({tmps[0]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader )
            ->record<kp::OpAlgoDispatch>(alg_3)
            ->record<kp::OpMemoryBarrier>({tmps[1]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader );            

        // t2 = t1 / 4        
        seq->record<kp::OpAlgoDispatch>(alg_div_4)
            ->record<kp::OpMemoryBarrier>({tmps[2]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader );            

        // t3 = MA(t2, 20)
        // t4 = MA(t2, 40)
        seq->record<kp::OpAlgoDispatch>(alg_ma_20)
            ->record<kp::OpAlgoDispatch>(alg_ma_40)
            ->record<kp::OpMemoryBarrier>({tmps[3], tmps[4]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader );
        
        // t5 = t3 - t4        
        seq->record<kp::OpAlgoDispatch>(alg_sub)
            ->record<kp::OpMemoryBarrier>({tmps[5]},
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferWrite,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer );
        
        // result = t5
        // GPU --> CPU
        seq->record<kp::OpTensorCopy>({tmps[5], Result})
            ->record<kp::OpTensorSyncLocal>({Result});
        
        seq->eval();

        memcpy( res.data(), Result->data(), size );
    }

    auto end = std::chrono::high_resolution_clock::now();

    return {std::chrono::duration<double>(end - start).count(), res};
}

void cpu_MA(const std::vector<float> &in, size_t MA, std::vector<float> &out)
{
    float acc = 0.f;
    size_t len = in.size();

    for( size_t i = 0; i < std::min(MA, len); ++i )
    {
        acc += in[i];
        out[i] = acc / (float)(i+1);
    }

    for( size_t i = MA; i < len; ++i )
    {
        acc += in[i];
        acc -= in[i-MA];
        out[i] = acc / MA;
    }
}

std::pair<double, std::vector<float>> time_cpu(size_t size, size_t tests)
{
    auto data = get_data(size);
    auto Open = data,
         High = data,
         Low = data,
         Close = data;
        
    std::vector<float> Result(size);
    
    std::vector<std::vector<float>> tmps;

    for( int i = 0; i < 7; ++i )
        tmps.push_back( std::vector<float>(size) );

    std::vector<float> res(size);

    auto start = std::chrono::high_resolution_clock::now();

    for( size_t i = 0; i < tests; ++i )
    {
        Open = data;
        High = data;
        Low = data;
        Close = data;

        // t1 = ((Open + High) + Low) + Close
        for( size_t i = 0; i < size; i++ )
            tmps[6][i] = Open[i] + High[i];
        
        for( size_t i = 0; i < size; i++ )
            tmps[0][i] = tmps[6][i] + Low[i];
        
        for( size_t i = 0; i < size; i++ )
            tmps[1][i] = tmps[0][i] + Close[i];
        
        // t2 = t1 / 4
        for( size_t i = 0; i < size; ++i )
            tmps[2][i] = tmps[1][i] / 4.f;
        
        // t3 = MA(t2, 20)
        // t4 = MA(t2, 40)
        cpu_MA(tmps[2], 20, tmps[3]);
        cpu_MA(tmps[2], 40, tmps[4]);

        // t5 = t3 - t4
        for( size_t i = 0; i < size; ++i )
            tmps[5][i] = tmps[3][i] - tmps[4][i];
        
        // result = t5
        Result = tmps[5];

        res = Result;
    }

    auto end = std::chrono::high_resolution_clock::now();

    return {std::chrono::duration<double>(end - start).count(), res};
}

int main()
{
    const size_t tests = 1;
    std::cout << "test reps: " << tests << "\n";

    for( size_t s : {10, 100, 1000, 10000, 100000, 1000000} )
    {
        s *= 1024;
        std::cout << s << " test: \n";

        auto [t_gpu, r_gpu] = time_gpu(s, tests);
        std::cout << "gpu: " << t_gpu << " (" << (t_gpu / tests)*1000 << "ms per test)\n";

        auto [t_cpu, r_cpu] = time_cpu(s, tests);
        std::cout << "cpu: " << t_cpu << " (" << (t_cpu / tests)*1000 << "ms per test)\n";

        if( r_gpu != r_cpu )
            std::cout << "different implementations result in different results!\n";

        std::cout << "\n";
    }
}
