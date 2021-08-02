// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "learner.h"
#include "parse_example.h"
#include "reduction_stack.h"

#include "io/logger.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace VW
{
namespace DELETE_FTR
{
struct feature_data
{
  std::string ftr_names;
  size_t call_count = 0;
};

template <bool is_learn, typename T, typename E>
void predict_or_learn(feature_data& data, T& base, E& ec)
{
  if (is_learn)
  {
    data.call_count++;
    base.learn(ec);

    // TODO: How to call hash from the reduction.
    // TODO: word_hash = (_p->hasher(feature_name.begin(), feature_name.length(), _channel_hash) & _parse_mask);
    // TODO: Which hash needs to be deleted. Go to the example and find that hash.
    // TODO: Use the namespace to get the namespace-feature dictionary lists and then delete? <cb_explore_adf_common>
  }
  else
  {
    // TODO: Predict with and without the feature for comparison side-by-side

    data.call_count++;
    base.predict(ec);
  }
}

VW::LEARNER::base_learner* delete_ftr_setup(VW::config::options_i& options, vw& all)
{
  // TODO: Setup
  auto data = scoped_calloc_or_throw<feature_data>();

  option_group_definition new_options("Delete features");
  new_options.add(make_option("del_ftr", data->ftr_names).help("Specify features to delete."));

  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  base_learner* base_learn = setup_base(options, all);

  if (base_learn->is_multiline)
  {
    learner<feature_data, multi_ex>* ret = &init_learner(data, as_multiline(base_learn),
        predict_or_learn<true, multi_learner, multi_ex>, predict_or_learn<false, multi_learner, multi_ex>, 1,
        base_learn->pred_type, all.get_setupfn_name(delete_ftr_setup), base_learn->learn_returns_prediction);

    return make_base(*ret);
  }
  else
  {
    learner<feature_data, example>* ret = &init_learner(data, as_singleline(base_learn),
        predict_or_learn<true, single_learner, example>, predict_or_learn<false, single_learner, example>, 1,
        base_learn->pred_type, all.get_setupfn_name(delete_ftr_setup), base_learn->learn_returns_prediction);

    return make_base(*ret);
  }
}

}  // namespace DELETE_FTR
}  // namespace VW
