#pragma once

#include <string>
#include <memory>

namespace checkers {

class RankLogger;

void sync_or_throw(const std::string& stage,
                   size_t index,
                   const std::string& name,
                   std::shared_ptr<RankLogger> logger);

} // namespace checkers