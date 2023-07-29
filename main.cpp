#include "cpu_ops.hpp"

#include "kompute/Kompute.hpp"

#include <iostream>
#include <memory>
#include <vector>
#include <math.h>
#include <chrono>
#include <stdexcept>

static std::vector<float> get_data(size_t size)
{
    std::vector<float> dat(size);

    float q = 0;
    for( float &f : dat )
        f = (q += 1);

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

kp::Manager mgr;

std::pair<double, std::vector<float>> time_gpu(size_t size, size_t tests)
{
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

    constexpr bool debug = false;

    for( int i = 0; i < 7; ++i )
        tmps.push_back( mgr.tensor(cst, debug ? kp::Tensor::TensorTypes::eDevice : kp::Tensor::TensorTypes::eStorage) );

    auto gpu_sum = get_spirv("sum.comp.spv");
    auto gpu_div_const = get_spirv("div_const.comp.spv");
    auto gpu_MA_const = get_spirv("MA_const.comp.spv");
    auto gpu_sub = get_spirv("sub.comp.spv");

    const kp::Workgroup workgroup{ static_cast<uint32_t>( (size + 63) / 64 ), 1, 1 };
    std::vector<uint32_t> size_push{{ static_cast<uint32_t>(size) }};

    auto alg_1 = mgr.algorithm({Open, High, tmps[6]}, gpu_sum, workgroup, {}, size_push);
    auto alg_2 = mgr.algorithm({tmps[6], Low, tmps[0]}, gpu_sum, workgroup, {}, size_push);
    auto alg_3 = mgr.algorithm({tmps[0], Close, tmps[1]}, gpu_sum, workgroup, {}, size_push);

    auto alg_div_4 = mgr.algorithm({tmps[1], tmps[2]}, gpu_div_const, workgroup, {4.f}, size_push);

    auto alg_ma_20 = mgr.algorithm<uint32_t,uint32_t>({tmps[2], tmps[3]}, gpu_MA_const, workgroup, {20}, size_push );
    auto alg_ma_40 = mgr.algorithm<uint32_t,uint32_t>({tmps[2], tmps[4]}, gpu_MA_const, workgroup, {40}, size_push );

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
        
        // Result = t5
        // GPU --> CPU
        seq->record<kp::OpTensorCopy>({tmps[5], Result})
            ->record<kp::OpTensorSyncLocal>({Result});
        
        if( debug ) seq->record<kp::OpTensorSyncLocal>(tmps);
        
        seq->eval();
    }

    auto end = std::chrono::high_resolution_clock::now();

    if( debug )
    {
        for( size_t i = 0; i < tmps.size(); i++ )
        {
            std::cout << "tmps[" << i << "]: {";
            for( float f : std::dynamic_pointer_cast<kp::TensorT<float>>(tmps[i])->vector() )
                std::cout << f << ", ";
            std::cout << "}\n";
        }

        std::cout << "Result: {";
        for( float f : Result->vector() )
            std::cout << f << ", ";
        std::cout << "}\n";
    }

    return {std::chrono::duration<double>(end - start).count(), Result->vector()};
}

std::pair<double, std::vector<float>> time_gpu_opt( size_t size, size_t tests )
{
    auto data = get_data( size );

    const kp::Constants cst( size );

    const auto input_type = kp::Tensor::TensorTypes::eHost;
    const auto output_type = kp::Tensor::TensorTypes::eDevice;

    auto Open = mgr.tensor( cst, input_type );
    auto High = mgr.tensor( cst, input_type );
    auto Low = mgr.tensor( cst, input_type );
    auto Close = mgr.tensor( cst, input_type );
    auto Result = mgr.tensor( cst, output_type );

    std::vector<std::shared_ptr<kp::Tensor>> tmps;

    constexpr bool debug = false;

    for( int i = 0; i < 3; ++i )
        tmps.push_back( mgr.tensor( cst, debug ? kp::Tensor::TensorTypes::eDevice : kp::Tensor::TensorTypes::eStorage ) );

    auto gpu_sum4_div_const = get_spirv( "sum4_div_const.comp.spv" );
    auto gpu_MA_const = get_spirv( "MA_const.comp.spv" );
    auto gpu_sub = get_spirv( "sub.comp.spv" );

    const kp::Workgroup workgroup{ static_cast<uint32_t>( ( size + 63 ) / 64 ), 1, 1 };
    std::vector<uint32_t> size_push{{ static_cast<uint32_t>( size ) }};

    auto alg_sum4_div_4 = mgr.algorithm( { Open, High, Low, Close, tmps[ 0 ] }, gpu_sum4_div_const, workgroup, { 4.f }, size_push );

    auto alg_ma_20 = mgr.algorithm<uint32_t, uint32_t>( { tmps[ 0 ], tmps[ 1 ] }, gpu_MA_const, workgroup, { 20 }, size_push );
    auto alg_ma_40 = mgr.algorithm<uint32_t, uint32_t>( { tmps[ 0 ], tmps[ 2 ] }, gpu_MA_const, workgroup, { 40 }, size_push );

    auto alg_sub = mgr.algorithm( { tmps[ 1 ], tmps[ 2 ], Result }, gpu_sub, workgroup, {}, size_push );

    auto seq = mgr.sequence();

    auto start = std::chrono::high_resolution_clock::now();

    for( size_t i = 0; i < tests; ++i )
    {
        Open->setData( data );
        High->setData( data );
        Low->setData( data );
        Close->setData( data );

        // CPU --> GPU
        seq->record<kp::OpTensorSyncDevice>( { Open, High, Low, Close } );

        // t0 = (Open + High + Low + Close) / 4        
        seq->record<kp::OpAlgoDispatch>( alg_sum4_div_4 )
            ->record<kp::OpMemoryBarrier>( { tmps[ 0 ] },
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader );

        // t1 = MA(t0, 20)
        // t2 = MA(t0, 40)
        seq->record<kp::OpAlgoDispatch>( alg_ma_20 )
            ->record<kp::OpAlgoDispatch>( alg_ma_40 )
            ->record<kp::OpMemoryBarrier>( { tmps[ 1 ], tmps[ 2 ] },
                vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
                vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader );

        // Result = t1 - t2
        // GPU --> CPU
        seq->record<kp::OpAlgoDispatch>( alg_sub )
            ->record<kp::OpTensorSyncLocal>( { Result } );

        if( debug ) seq->record<kp::OpTensorSyncLocal>( tmps );

        seq->eval();
    }

    auto end = std::chrono::high_resolution_clock::now();

    if( debug )
    {
        for( size_t i = 0; i < tmps.size(); i++ )
        {
            std::cout << "tmps[" << i << "]: {";
            for( float f : std::dynamic_pointer_cast<kp::TensorT<float>>( tmps[ i ] )->vector() )
                std::cout << f << ", ";
            std::cout << "}\n";
        }

        std::cout << "Result: {";
        for( float f : Result->vector() )
            std::cout << f << ", ";
        std::cout << "}\n";
    }

    return { std::chrono::duration<double>( end - start ).count(), Result->vector() };
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
        cpu_add(Open, High, tmps[6]);
        cpu_add(tmps[6], Low, tmps[0]);
        cpu_add(tmps[0], Close, tmps[1]);

        // t2 = t1 / 4
        cpu_div_const(tmps[1], 4.f, tmps[2]);

        // t3 = MA(t2, 20)
        // t4 = MA(t2, 40)
        cpu_MA_const(tmps[2], 20, tmps[3]);
        cpu_MA_const(tmps[2], 40, tmps[4]);

        // t5 = t3 - t4
        cpu_sub(tmps[3], tmps[4], tmps[5]);

        // Result = t5
        cpu_cpy(tmps[5], Result);

        res = Result;
    }

    auto end = std::chrono::high_resolution_clock::now();

    return {std::chrono::duration<double>(end - start).count(), res};
}

bool cmp_res(float A, float B) 
{
   float diff = A - B;
   return (diff < 1e-3f) && (-diff < 1e-3f);
}

int main()
{
    const size_t tests = 1;
    std::cout << "test reps: " << tests << "\n";
    
    for( size_t s : {10, 100, 1000, 10000} )
    {
        s *= 1000;
        std::cout << s << " test: \n";

        auto [t_gpu, r_gpu] = time_gpu(s, tests);
        std::cout << "gpu: " << t_gpu << " (" << (t_gpu / tests)*1000 << "ms per test)\n";

        auto [t_cpu, r_cpu] = time_cpu(s, tests);
        std::cout << "cpu: " << t_cpu << " (" << (t_cpu / tests)*1000 << "ms per test)\n";

        auto [t_gpu_o, r_gpu_o] = time_gpu_opt( s, tests );
        std::cout << "gpu opt: " << t_gpu_o << " (" << ( t_gpu_o / tests ) * 1000 << "ms per test)\n";

        for( size_t i = 0; i < s; i++ )
            if( !cmp_res( r_gpu[i], r_cpu[i] ) )
            {
                std::cout << "different implementations result in different results! (gpu-cpu)\n";
                break;
            }
        
        for( size_t i = 0; i < s; i++ )
            if( !cmp_res( r_gpu[ i ], r_gpu_o[ i ] ) )
            {
                std::cout << "different implementations result in different results! (gpu-gpu opt)\n";
                break;
            }
        
        std::cout << "\n";
    }
}
