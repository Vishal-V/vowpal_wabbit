// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#pragma once

#include "reductions_fwd.h"

namespace VW
{
namespace DELETE_FTR
{
VW::LEARNER::base_learner* delete_ftr_setup(VW::config::options_i& options, vw& all);
}
}  // namespace VW
