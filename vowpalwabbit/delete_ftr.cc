// Copyright (c) by respective owners including Yahoo!, Microsoft, and
// individual contributors. All rights reserved. Released under a BSD (revised)
// license as described in the file LICENSE.

#include <vector>
#include "reductions.h"
#include "learner.h"
#include "parse_example.h"
#include "parser.h"
#include "example.h"
#include "feature_group.h"
#include <boost/type_index.hpp>

#include "io/logger.h"

using namespace VW::config;
using namespace VW::LEARNER;

namespace VW
{
namespace DELETE_FTR
{
struct feature_data
{
  vw* all;
  std::string namespace_name = " ";
  std::string ftr_names;
  // example buffer_ec;
  example* modify;
  size_t num_ftr = 0;
  size_t manip_flag = 0;
  size_t mod_flag = 0;
  size_t namespace_hash;
  size_t ftr_hash;
  double value = 1;
  namespace_index index = ' ';
  size_t mod_hash;
  size_t rename_flag = 0;
  std::string mod_ftr_name;
};

inline void delete_feature(feature* ftr) { return_features(ftr); }

inline void delete_feature(example& ec, namespace_index index, size_t feature_hash)
{
  if (ec.feature_space[index].indicies[0] == feature_hash)  // TODO
  {
    // ec.feature_space[index].indicies[0] = feature_hash;
    // ec.feature_space[index].values[0] = value;
    VW::io::logger::log_warn(
        "Value modified for feature_hash {} to {}", feature_hash, ec.feature_space[index].values[0]);
  }
}

inline void modify_feature(example& ec, feature_data data, int& idx_ret)
{
  // VW::io::logger::log_warn("Features: {}, {}, {}", data.ftr_hash, fs.indicies[0], fs.values[0]);
  for (unsigned int idx = 0; idx < ec.feature_space[data.index].indicies.size(); idx++)
  {
    if (ec.feature_space[data.index].indicies[idx] == data.ftr_hash)
    {
      if (data.rename_flag)
      {
        ec.feature_space[data.index].indicies[idx] = data.mod_hash;
        VW::io::logger::log_warn(
            "Feature renamed to {} with hash {}", data.mod_ftr_name, ec.feature_space[data.index].indicies[idx]);
      }
      if (ec.feature_space[data.index].values[idx] != data.value && data.mod_flag)
      {
        ec.feature_space[data.index].values[idx] = data.value;
        VW::io::logger::log_warn("Value modified for data.ftr_hash {} to {}",
            ec.feature_space[data.index].indicies[idx], ec.feature_space[data.index].values[idx]);
      }
      idx_ret = idx;
    }
  }
}

// void del_example_namespace(example& ec, namespace_index ns, features& fs)
// {
//   // print_update is called after this del_example_namespace,
//   // so we need to keep the ec.num_features correct,
//   // so shared features are included in the reported number of "current features"
//   // ec.num_features -= numf;
//   features& del_target = ec.feature_space[static_cast<size_t>(ns)];
//   assert(del_target.size() >= fs.size());
//   assert(ec.indices.size() > 0);
//   if (ec.indices.back() == ns && ec.feature_space[static_cast<size_t>(ns)].size() == fs.size())
//   ec.indices.pop_back(); ec.reset_total_sum_feat_sq(); ec.num_features -= fs.size();
//   del_target.truncate_to(del_target.size() - fs.size());
//   del_target.sum_feat_sq -= fs.sum_feat_sq;
// }

inline void check_modify_feature(example& ec, namespace_index index, size_t feature_hash, int idx)
{
  if (ec.feature_space[index].indicies[idx] == feature_hash)
  {
    VW::io::logger::log_warn(
        "Check: modification of feature_hash {} to {}", feature_hash, ec.feature_space[index].values[idx]);
  }
}

void manipulate_features(feature_data& data, example& ec, void (*fn)(feature* ftr) = nullptr)
{
  // size_t ftr_num = (&ec)->num_features;  // get_feature_number(&ec);
  // data.num_ftr = ftr_num;
  // feature* ftr = get_features(*(data.all), &ec, (&data)->num_ftr);

  int idx = 0;
  for (namespace_index c : ec.indices)
  {
    data.index = c;
    if (c == (namespace_index)data.namespace_name[0]) break;
  }

  data.namespace_hash = hash_space(*(data.all), data.namespace_name);
  data.ftr_hash = hash_feature(*(data.all), data.ftr_names, data.namespace_hash);
  data.mod_hash = hash_feature(*(data.all), data.mod_ftr_name, data.namespace_hash);

  uint64_t multiplier = static_cast<uint64_t>(data.all->wpp) << data.all->weights.stride_shift();
  data.ftr_hash *= multiplier;
  data.mod_hash *= multiplier;

  modify_feature(ec, data, idx);
  // data.index, data.ftr_hash * multiplier, idx, || data.value, data.mod_flag, data.rename_flag, data.mod_hash);
  if (data.mod_flag) check_modify_feature(ec, data.index, data.ftr_hash, idx);

  feature* ftr = nullptr;        // Modify after test
  if (*fn) data.manip_flag = 1;  // Modify after test
  if (data.manip_flag)
  {
    // delete_feature((ftr + 1));
    return;  // data.manip_ec;
  }
  else
  {
    (*fn)(ftr);  // (*fn)(ftr, hash_val);
  }
  // TODO: match feature with hash and get the feature pointer for example
  // size_t get_feature_hash(std::string ftr_name) in example.cc
  // int check_feature_hash_exists(size_t hash) in example.cc
  // feature* get_feature_with_hash(size_t hash) in example.cc
  // TODO: Hash and add the feature to the example after manipulation
}

template <bool is_learn, typename T, typename E>
void predict_or_learn(feature_data& data, T& base, E& ec)
{
  // if (data.all->options->was_supplied("del_ftr"))
  // {
  example* copy_ec = alloc_examples(1);
  copy_example_data_with_label(copy_ec, &ec);
  data.modify = copy_ec;
  manipulate_features(data, *data.modify, delete_feature);
  if (is_learn)
  {
    if (!data.manip_flag) { base.learn(ec); }
    else
    {
      base.learn(*data.modify);
      ec.pred.scalar = std::move(data.modify->pred.scalar);
    }

    // TODO: test_case for hashing and deleting
    // TODO: Design a class structure
  }
  else
  {
    if (!data.manip_flag) { base.predict(ec); }
    else
    {
      base.predict(*data.modify);
      ec.pred.scalar = std::move(data.modify->pred.scalar);
    }
  }
  // }
  // else
  // {
  //   if (is_learn)
  //     base.learn(ec);
  //   else
  //     base.predict(ec);
  // }
}

void finish_example(vw& all, feature_data& data, example&) { output_and_account_example(all, *data.modify); }

VW::LEARNER::base_learner* delete_ftr_setup(setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();
  vw& all = *stack_builder.get_all_pointer();
  auto data = scoped_calloc_or_throw<feature_data>();
  bool manip_ftr = false;
  // TODO: Option to specify the namespace from which to delete
  option_group_definition new_options("Delete features");
  new_options.add(make_option("ftr_manip", manip_ftr).necessary().help("Manipualte the specified."));
  new_options.add(make_option("del_ftr", data->ftr_names).help("Specify features to delete."));
  new_options.add(
      make_option("del_ftr_nms", data->namespace_name).help("Specify namespace for the feature to be modified."));
  new_options.add(make_option("mod_val", data->value).help("Specify the modified value for the feature."));
  new_options.add(make_option("rename_ftr", data->mod_ftr_name).help("Rename the feature."));
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;
  // options.add_and_parse(new_options);
  data->all = &all;

  if (all.options->was_supplied("del_ftr"))
    VW::io::logger::log_warn("Setup options for deleting feature name: {}", data->ftr_names);
  if (all.options->was_supplied("mod_val")) data->mod_flag = 1;
  if (all.options->was_supplied("rename_ftr")) data->rename_flag = 1;

  base_learner* base_learn = stack_builder.setup_base_learner();

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
  ret->set_finish_example(finish_example);
  return make_base(*ret);
  // }
}

}  // namespace DELETE_FTR
}  // namespace VW
