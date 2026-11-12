// Included by ResourceMaterializationTests.cpp to use the same standalone IR fixture.
#include <chrono>
#include <cstring>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;

Program DrawResourceProgram(uint32_t buffer_count) {
  Program program;
  program.stage = Libs::Graphics::ShaderType::Compute;
  program.user_data_count = 3;
  program.srt_plan_complete = true;
  program.resource_tracking_complete = true;
  auto &block = AddValueBlock(program);
  const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(op, args));
  };
  MemoryInfo memory;
  memory.kind = ResourceKind::ScalarAddress;
  memory.planning_only = true;
  program.memory_info.push_back(memory);
  const auto read = [&](Value base, Value offset) {
    const auto value = emit(ValueOpcode::LoadAddressU32,
                            {base, offset, Value(0u), Value(true)});
    value.Instruction()->SetFlags(MemoryFlags{.index = 0});
    const auto slot = static_cast<uint32_t>(program.srt_reads.size());
    program.srt_reads.push_back({value, slot});
    return value;
  };
  const auto low = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
  const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(1))});
  const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
  const auto table_low = read(root, Value(0u));
  const auto table_high = read(root, Value(4u));
  const auto table = emit(ValueOpcode::GetAddressResource, {table_low, table_high});
  const auto offset = emit(ValueOpcode::GetUserData, {Value(ScalarReg(2))});
  const auto srt = emit(ValueOpcode::GetSrtResource, {});
  for (uint32_t index = 0; index < buffer_count; ++index) {
    DescriptorSource source;
    source.dword_count = 4;
    for (uint32_t word = 0; word < 4; ++word) {
      const auto byte_offset = emit(ValueOpcode::IAdd32,
                                     {offset, Value((index * 4u + word) * 4u)});
      const auto slot = static_cast<uint32_t>(program.srt_reads.size());
      read(table, byte_offset);
      source.dwords[word] = emit(ValueOpcode::ReadConst, {srt, Value(slot)});
    }
    program.info.buffers.push_back({.source = index});
    program.descriptor_sources.push_back(source);
  }
  return program;
}

std::array<uint32_t, 2> SrtPointer(const void *pointer) {
  const auto address = reinterpret_cast<uint64_t>(pointer);
  return {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u)};
}

void TestDrawResourceWalkRefresh() {
  for (const uint32_t count : {1u, 2u, 3u, 8u, 16u, 31u, 32u, 64u}) {
    const auto plan = Extract(DrawResourceProgram(count));
    std::vector<uint32_t> first(count * 4u + 4u), second(first.size());
    for (uint32_t i = 0; i < first.size(); ++i) {
      first[i] = 0x100u + i;
      second[i] = 0x800u + i;
    }
    auto table = SrtPointer(first.data());
    auto other_table = SrtPointer(second.data());
    auto root = SrtPointer(table.data());
    std::array<uint32_t, 3> user_data{root[0], root[1], 0u};
    const SrtRuntime runtime{.user_data = user_data};
    const auto check = [&](const std::vector<uint32_t> &expected, uint32_t offset) {
      std::vector<uint32_t> flat;
      std::vector<std::pair<uint64_t, uint64_t>> reads;
      Check(WalkSrt(plan, runtime, flat, &reads), "draw resource walk failed");
      Check(flat.size() == count * 4u + 2u && reads.size() == flat.size(),
            "shared operands changed scalar reads or their captures");
      const auto root_address = (uint64_t{user_data[1] & 0xffffu} << 32u) |
                                (user_data[0] & ~3u);
      Check(reads[0] == std::pair<uint64_t, uint64_t>{root_address, 4u} &&
                reads[1] == std::pair<uint64_t, uint64_t>{root_address + 4u, 4u},
            "a walk captured an old root table");
      for (uint32_t i = 0; i < count; ++i) {
        DescriptorValue descriptor;
        Check(WalkedDescriptor(plan, i, descriptor) && descriptor.dword_count == 4u,
              "draw descriptor was unknown");
        for (uint32_t word = 0; word < 4u; ++word) {
          const auto index = i * 4u + word;
          Check(descriptor.dwords[word] == expected[index + offset] &&
                    flat[index + 2u] == expected[index + offset],
                "a new walk reused an old descriptor operand");
          Check(reads[index + 2u] == std::pair<uint64_t, uint64_t>{
                    reinterpret_cast<uint64_t>(expected.data() + index + offset), 4u},
                "a walk captured an old descriptor address");
        }
      }
    };
    check(first, 0u);
    // Each descriptor word changes independently at the same guest address.
    for (auto &word : first) {
      word ^= 0x12340000u;
      check(first, 0u);
    }
    table = SrtPointer(second.data());
    check(second, 0u);
    root = SrtPointer(other_table.data());
    user_data[0] = root[0] | 3u;
    user_data[1] = root[1] | 0xabcd0000u;
    check(second, 0u);
    user_data[2] = 19u; // Scalar reads clear the low two address bits.
    check(second, 4u);
    other_table = SrtPointer(first.data());
    check(first, 4u);
    std::vector<uint32_t> flat;
    Check(!WalkSrt(plan, {.user_data = std::span(user_data).first(2)}, flat, nullptr),
          "a walk accepted a missing user-data input");
  }

  Program source;
  source.srt_plan_complete = true;
  source.resource_tracking_complete = true;
  source.user_data_count = 1;
  auto &block = AddValueBlock(source);
  const auto emit = [&](ValueOpcode op, std::initializer_list<Value> args) {
    return Value(&block.AppendNewInst(op, args));
  };
  const auto base = emit(ValueOpcode::GetShaderBase, {});
  const auto low = emit(ValueOpcode::CompositeExtractU64, {base, Value(0u)});
  const auto high = emit(ValueOpcode::CompositeExtractU64, {base, Value(1u)});
  const auto offset = emit(ValueOpcode::GetUserData, {Value(ScalarReg(0))});
  DescriptorSource descriptor;
  descriptor.dword_count = 4;
  descriptor.dwords = {emit(ValueOpcode::IAdd32, {low, offset}), high,
                        Value(4u), Value(0u)};
  source.descriptor_sources.push_back(descriptor);
  source.info.buffers.push_back({.source = 0});
  const auto plan = Extract(source);
  std::array<uint32_t, 1> user_data{7u};
  std::vector<uint32_t> flat;
  for (const uint64_t shader_base : {0x123456780000ull, 0x56789abc0000ull, 0x123456780000ull}) {
    for (const uint32_t value : {7u, 23u, 7u}) {
      user_data[0] = value;
      DescriptorValue current;
      Check(WalkSrt(plan, {.user_data = user_data, .shader_base = shader_base}, flat, nullptr) &&
                WalkedDescriptor(plan, 0u, current) &&
                current.dwords[0] == static_cast<uint32_t>(shader_base) + value &&
                current.dwords[1] == static_cast<uint32_t>(shader_base >> 32u),
            "a walk reused a prior shader base or user-data value");
    }
  }
}

void TestSharedOperandRefresh() {
  // Different graph sizes exercise short clears and the large-clear fallback.
  for (uint32_t depth = 0; depth < 270u; ++depth) {
    Program program;
    program.srt_plan_complete = true;
    program.resource_tracking_complete = true;
    program.user_data_count = 1;
    auto &block = AddValueBlock(program);
    auto value = Value(&block.AppendNewInst(ValueOpcode::GetUserData, {Value(ScalarReg(0))}));
    for (uint32_t index = 0; index < depth; ++index) {
      value = Value(&block.AppendNewInst(ValueOpcode::IAdd32, {value, Value(1u)}));
    }
    DescriptorSource source;
    source.dword_count = 4;
    source.dwords.fill(value);
    program.descriptor_sources.push_back(source);
    program.info.buffers.push_back({.source = 0});
    const auto plan = Extract(program);
    std::array<uint32_t, 1> user_data{};
    std::vector<uint32_t> flat;
    for (const uint32_t input : {0u, 17u, UINT32_MAX, 0u}) {
      user_data[0] = input;
      DescriptorValue current;
      Check(WalkSrt(plan, {.user_data = user_data}, flat, nullptr) &&
                WalkedDescriptor(plan, 0u, current),
            "shared arithmetic operand was unknown");
      for (uint32_t word = 0; word < 4u; ++word) {
        Check(current.dwords[word] == input + depth,
              "shared arithmetic operand kept a prior draw's value");
      }
    }
  }
}

void BenchmarkDrawResources() {
  using Clock = std::chrono::steady_clock;
  constexpr uint32_t iterations = 100000;
  for (const uint32_t count : {1u, 8u, 32u}) {
    std::vector<uint32_t> words(count * 4u);
    for (uint32_t i = 0; i < count; ++i) {
      words[i * 4u] = 0x8000u + i * 0x100u;
      words[i * 4u + 1u] = 4u << 16u;
      words[i * 4u + 2u] = 128u;
      words[i * 4u + 3u] = 3u << 28u;
    }
    const auto table = SrtPointer(words.data());
    const auto root = SrtPointer(table.data());
    const std::array<uint32_t, 3> user_data{root[0], root[1], 0u};
    const auto plan = Extract(DrawResourceProgram(count));
    std::printf("SrtDrawBenchmarkSize buffers=%u code_bytes=%zu scratch_bytes=%zu\n",
                count, plan.srt_code->getSize(), plan.srt_scratch.size() * sizeof(uint64_t));
    const SrtRuntime runtime{.user_data = user_data};
    std::vector<uint32_t> flat;
    ResourceSnapshot snapshot;
    ResourceSpecialization specialization;
    Check(MaterializeResources(plan, runtime, snapshot, specialization),
          "benchmark resource materialization failed");
    for (uint32_t repeat = 0; repeat < 5; ++repeat) {
      for (const bool materialize : {false, true}) {
        uint64_t checksum = 0;
        const auto start = Clock::now();
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
          // Change live descriptor data. No input or result can be cached across draws.
          words[0] = 0x8000u + (iteration & 255u) * 4u;
          if (materialize) {
            Check(MaterializeResources(plan, runtime, snapshot, specialization),
                  "benchmark materialization failed");
            checksum += snapshot.buffers[0].dwords[0];
          } else {
            Check(WalkSrt(plan, runtime, flat, nullptr), "benchmark walk failed");
            checksum += flat[2];
          }
        }
        const auto elapsed = std::chrono::duration<double, std::nano>(Clock::now() - start).count();
        std::printf("SrtDrawBenchmark buffers=%u mode=%s repeat=%u iterations=%u ns/draw=%.2f checksum=%llu\n",
                    count, materialize ? "materialize" : "walk", repeat, iterations,
                    elapsed / iterations, static_cast<unsigned long long>(checksum));
      }
    }
  }
}

} // namespace
