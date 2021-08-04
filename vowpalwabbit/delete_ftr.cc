// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include "reductions.h"
#include "learner.h"
#include "parse_example.h"
#include "parser.h"
#include "example.h"

#include "io/logger.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace VW
{
namespace DELETE_FTR
{
struct feature_data
{
  std::string namespace_name;
  std::string ftr_names;
  example* manip_ec;
  example* non_manip;
  size_t num_ftr = 0;
  size_t manip_flag = 0;
};

// Maybe return feature*? void (*fn)(feature* ftr, size_t hash)
void manipulate_features(feature_data& data, example& ec, void (*fn)(feature* ftr) = nullptr)
{
  size_t ftr_num = (&ec)->num_features;  // get_feature_number(&ec);
  data.num_ftr = ftr_num;
  // TODO: match feature with hash and get the feature pointer for example
  // size_t get_feature_hash(std::string ftr_name) in example.cc
  // int check_feature_hash_exists(size_t hash) in example.cc
  // feature* get_feature_with_hash(size_t hash) in example.cc
  // TODO: Hash and add the feature to the example after manipulation
  feature* ftr = nullptr;        // Modify after test
  if (*fn) data.manip_flag = 1;  // Modify after test
  if (data.manip_flag)
    return;  // data.manip_ec;
  else
  {
    (*fn)(ftr);  // (*fn)(ftr, hash_val);
  }
}

void delete_feature(feature* ftr) { return_features(ftr); }

template <bool is_learn, typename T, typename E>
void predict_or_learn(feature_data& data, T& base, E& ec)
{
  if (is_learn)
  {
    data.namespace_name = " ";  // Temporary hard-code
    data.ftr_names = "b";       // Temporary hard-code
    // feature* get_features(vw& all, example* ec, size_t& feature_number);
    // VW::io::logger::errlog_warn("Feature to be deleted: {} from total features.", data.ftr_names);

    example* copy_ec = alloc_examples(1);
    copy_example_data_with_label(copy_ec, &ec);
    data.non_manip = copy_ec;
    manipulate_features(data, ec, delete_feature);
    if (data.manip_flag) { base.learn(ec); }
    else
    {
      base.learn(*data.non_manip);
    }
    // base.learn(ec);

    // TODO: test_case for hashing and deleting
    // TODO: Design a class structure
    // TODO: How to call hash from the reduction.
    // TODO: word_hash = (_p->hasher(feature_name.begin(), feature_name.length(), _channel_hash) & _parse_mask);
    // TODO: Which hash needs to be deleted. Go to the example and find that hash.
  }
  else
  {
    // TODO: Predict with and without the feature for comparison side-by-side
    base.predict(ec);
  }
}

VW::LEARNER::base_learner* delete_ftr_setup(VW::config::options_i& options, vw& all)
{
  auto data = scoped_calloc_or_throw<feature_data>();

  option_group_definition new_options("Delete features");
  new_options.add(make_option("del_ftr", data->ftr_names).help("Specify features to delete."));

  // if (!options.add_parse_and_check_necessary(new_options)) return nullptr;
  options.add_and_parse(new_options);

  base_learner* base_learn = setup_base(options, all);

  // TODO: Fix multiline after test
  // if (base_learn->is_multiline)
  // {
  //   learner<feature_data, multi_ex>* ret = &init_learner(data, as_multiline(base_learn),
  //       predict_or_learn<true, multi_learner, multi_ex>, predict_or_learn<false, multi_learner, multi_ex>, 1,
  //       base_learn->pred_type, all.get_setupfn_name(delete_ftr_setup), base_learn->learn_returns_prediction);

  //   return make_base(*ret);
  // }
  // else
  // {
  learner<feature_data, example>* ret = &init_learner(data, as_singleline(base_learn),
      predict_or_learn<true, single_learner, example>, predict_or_learn<false, single_learner, example>, 1,
      base_learn->pred_type, all.get_setupfn_name(delete_ftr_setup), base_learn->learn_returns_prediction);

  return make_base(*ret);
  // }
}

}  // namespace DELETE_FTR
}  // namespace VW
