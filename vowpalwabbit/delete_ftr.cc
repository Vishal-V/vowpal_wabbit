// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "parse_example.h"

#include "io/logger.h"

namespace VW
{
namespace DELETE_FTR
{
template <bool is_learn, typename T, typename E>
void predict_or_learn(VW::LEARNER::single_learner& base, example& ec)
{
  if (is_learn)
  {
    base.learn(ec);

    // TODO: How to call hash from the reduction.
    // TODO: word_hash = (_p->hasher(feature_name.begin(), feature_name.length(), _channel_hash) & _parse_mask);
    // TODO: This is the hash that should be deleted. Go to the example and find that hash.
  }
  else
  {
    // TODO: Predict with and without the feature for comparison side-by-side

    base.predict(ec);
  }
}

VW::LEARNER::base_learner* delete_ftr_setup(VW::config::options_i& options, vw& all)
{
  // TODO: Setup
  std::string s;
  option_group_definition new_options("Delete features");
  new_options.add(make_option("del_ftr", s)).help("Specify features to delete."));

  base_learner* base_learn = setup_base(options, all);
  if (base_learn->is_multiline)
  {
    auto ret = VW::LEARNER::make_no_data_reduction_learner(
        as_multiline(base_learn), predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(binary_setup))
                   .set_learn_returns_prediction(true)
                   .build();
  }
  else
  {
    auto ret = VW::LEARNER::make_no_data_reduction_learner(
        as_singleline(base_learn), predict_or_learn<true>, predict_or_learn<false>, all.get_setupfn_name(binary_setup))
                   .set_learn_returns_prediction(true)
                   .build();
  }

  return make_base(*ret);
}

}  // namespace DELETE_FTR
}  // namespace VW
