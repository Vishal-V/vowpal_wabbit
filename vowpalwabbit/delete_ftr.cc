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
  std::string mod_ftr_name;
  example* modify;
  size_t num_ftr = 0;
  size_t namespace_hash;
  size_t ftr_hash;
  double value = 1;
  namespace_index index = ' ';
  size_t mod_hash;
  double log_base = 2;
  double bin_thresh = 0;
  size_t manip_flag = false;
  size_t mod_flag = false;
  size_t rename_flag = false;
  size_t delete_flag = false;
  size_t binarize_flag = false;
  size_t log_flag = false;
  size_t audit_flag = false;
};

template <class ForwardIt, class ForwardItFloat, class ForwardItPair, class T>
ForwardIt remove_ftr(ForwardIt first_hash, ForwardIt last_hash, ForwardItFloat first_val, ForwardItFloat last_val,
    ForwardItPair first_audit, ForwardItPair last_audit, const T& value)
{
  first_hash = std::find(first_hash, last_hash, value);
  if (first_hash != last_hash)
  {
    auto j = first_val;
    auto k = first_audit;
    for (auto i = first_hash; ++i != last_hash && ++j != last_val && ++k != last_audit;)
    {
      if (!(*i == value))
      {
        *first_hash++ = std::move(*i);
        *first_val++ = std::move(*j);
        // (*first_audit).first = std::move((*k).first);
        // (*first_audit).second = std::move((*k).second);
        // *first_audit++;  // = std::move(*k);
      }
    }
  }
  return first_hash;
}

inline void delete_feature(example& ec, feature_data data)
{
  if (ec.indices.size() == 0) return;

  int len_ec = ec.feature_space[static_cast<size_t>(data.index)].indicies.size(), num_ftr_del = 0;
  for (int i = 0; i < len_ec; i++)
    if (ec.feature_space[static_cast<size_t>(data.index)].indicies[i] == data.ftr_hash) { num_ftr_del++; }
  for (int i = 0; i < len_ec; i++)
  {
    if (ec.feature_space[static_cast<size_t>(data.index)].indicies[i] == data.ftr_hash)
    {
      if (data.audit_flag)
      {
        unsigned int last_idx = 0, curr_idx = 0;
        while (curr_idx < ec.feature_space[static_cast<size_t>(data.index)].size())
        {
          if (ec.feature_space[static_cast<size_t>(data.index)].indicies[curr_idx] != data.ftr_hash)
          {
            ec.feature_space[static_cast<size_t>(data.index)].space_names[last_idx] =
                ec.feature_space[static_cast<size_t>(data.index)].space_names[curr_idx];
            last_idx++;
          }
          curr_idx++;
        }
      }

      remove_ftr(ec.feature_space[static_cast<size_t>(data.index)].indicies.begin(),
          ec.feature_space[static_cast<size_t>(data.index)].indicies.end(),
          ec.feature_space[static_cast<size_t>(data.index)].values.begin(),
          ec.feature_space[static_cast<size_t>(data.index)].values.end(),
          ec.feature_space[static_cast<size_t>(data.index)].space_names.begin(),
          ec.feature_space[static_cast<size_t>(data.index)].space_names.end(), data.ftr_hash);
    }
  }
  while (num_ftr_del)
  {
    if (ec.feature_space[static_cast<size_t>(data.index)].size() == 1)
    {
      assert(ec.feature_space[static_cast<size_t>(data.index)].indicies[0] == data.ftr_hash);
      VW::io::logger::log_warn("Deleting Namespace: {}", ec.indices.back());
      ec.indices.pop_back();
      ec.num_features--;
      num_ftr_del--;
      break;
    }
    else
    {
      VW::io::logger::log_warn(
          "Deleting Feature: {}", ec.feature_space[static_cast<size_t>(data.index)].indicies.back());
      ec.feature_space[static_cast<size_t>(data.index)].indicies.pop_back();
      ec.feature_space[static_cast<size_t>(data.index)].values.pop_back();
      // ec.feature_space[static_cast<size_t>(data.index)].space_names.pop_back();
      ec.num_features--;
      num_ftr_del--;
    }
  }
  // del_target.sum_feat_sq -= fs.sum_feat_sq;
  return;
}

inline void modify_feature(example& ec, feature_data data, int& idx_ret)
{
  for (unsigned int idx = 0; idx < ec.feature_space[data.index].indicies.size(); idx++)
  {
    if (ec.feature_space[static_cast<size_t>(data.index)].indicies[idx] == data.ftr_hash)
    {
      // TODO: Fix fs.sum_feat_sq
      if (data.delete_flag)
      {
        if (ec.indices.size() == 0) return;
        if (ec.indices.back() == data.index && ec.feature_space[static_cast<size_t>(data.index)].size() == 1)
        {
          assert(ec.feature_space[static_cast<size_t>(data.index)].indicies[0] == data.ftr_hash);
          VW::io::logger::log_warn("Deleting Namespace!");
          ec.indices.pop_back();
          ec.num_features--;
        }
        else
        {
          int len_ec = ec.feature_space[static_cast<size_t>(data.index)].size(), num_ftr_del = 0;
          for (int i = 0; i < len_ec; i++)
            if (ec.feature_space[static_cast<size_t>(data.index)].indicies[i] == data.ftr_hash) { num_ftr_del++; }
          for (int i = 0; i < len_ec; i++)
          {
            if (ec.feature_space[static_cast<size_t>(data.index)].indicies[i] == data.ftr_hash)
            {
              remove_ftr(ec.feature_space[static_cast<size_t>(data.index)].indicies.begin(),
                  ec.feature_space[static_cast<size_t>(data.index)].indicies.end(),
                  ec.feature_space[static_cast<size_t>(data.index)].values.begin(),
                  ec.feature_space[static_cast<size_t>(data.index)].values.end(),
                  ec.feature_space[static_cast<size_t>(data.index)].space_names.begin(),
                  ec.feature_space[static_cast<size_t>(data.index)].space_names.end(), data.ftr_hash);
            }
          }
          while (num_ftr_del)
          {
            if (ec.feature_space[static_cast<size_t>(data.index)].size() == 1)
            {
              assert(ec.feature_space[static_cast<size_t>(data.index)].indicies[0] == data.ftr_hash);
              VW::io::logger::log_warn("Deleting Namespace: {}", ec.indices.back());
              ec.indices.pop_back();
              ec.num_features--;
              num_ftr_del--;
              break;
            }
            else
            {
              VW::io::logger::log_warn(
                  "Deleting Feature: {}!", ec.feature_space[static_cast<size_t>(data.index)].indicies.back());
              ec.feature_space[static_cast<size_t>(data.index)].indicies.pop_back();
              ec.feature_space[static_cast<size_t>(data.index)].values.pop_back();
              ec.feature_space[static_cast<size_t>(data.index)].space_names.pop_back();
              ec.num_features--;
              num_ftr_del--;
            }
          }
        }
        // del_target.sum_feat_sq -= fs.sum_feat_sq;
        break;
      }
      if (data.rename_flag)
      {
        ec.feature_space[static_cast<size_t>(data.index)].indicies[idx] = data.mod_hash;
        VW::io::logger::log_warn("Feature renamed to {} with hash {}", data.mod_ftr_name,
            ec.feature_space[static_cast<size_t>(data.index)].indicies[idx]);
      }
      if (data.mod_flag)
      {
        // size_t val = ec.feature_space[static_cast<size_t>(data.index)].values[idx];
        // val *= val;
        ec.feature_space[static_cast<size_t>(data.index)].values[idx] = data.value;
        // ec.feature_space[static_cast<size_t>(data.index)].sum_feat_sq -= val;
        // ec.feature_space[static_cast<size_t>(data.index)].sum_feat_sq += data.value * data.value;
        VW::io::logger::log_warn("Value modified for data.ftr_hash {} to {}",
            ec.feature_space[static_cast<size_t>(data.index)].indicies[idx],
            ec.feature_space[static_cast<size_t>(data.index)].values[idx]);
      }
      else if (data.binarize_flag)
      {
        ec.feature_space[static_cast<size_t>(data.index)].values[idx] =
            (ec.feature_space[static_cast<size_t>(data.index)].values[idx] < data.bin_thresh) ? 0 : 1;
        VW::io::logger::log_warn("Value binarized with thresold {} for {} to {}", data.bin_thresh,
            ec.feature_space[static_cast<size_t>(data.index)].indicies[idx],
            ec.feature_space[static_cast<size_t>(data.index)].values[idx]);
      }
      else if (data.log_flag)
      {
        if (data.log_base == 1) { ec.feature_space[static_cast<size_t>(data.index)].values[idx] = 0; }
        else
          ec.feature_space[static_cast<size_t>(data.index)].values[idx] =
              (ec.feature_space[static_cast<size_t>(data.index)].values[idx] <= 0)
              ? 0
              : log(ec.feature_space[static_cast<size_t>(data.index)].values[idx]) / log(data.log_base);
        VW::io::logger::log_warn("Value manipulated to logarithm scale with base {} for {} to {}", data.log_base,
            ec.feature_space[static_cast<size_t>(data.index)].indicies[idx],
            ec.feature_space[static_cast<size_t>(data.index)].values[idx]);
      }
      idx_ret = idx;
    }
  }
}

inline void check_modify_feature(example& ec, namespace_index index, size_t feature_hash, int idx)
{
  if (ec.feature_space[index].indicies[idx] == feature_hash)
  {
    VW::io::logger::log_warn(
        "Check: modification of feature_hash {} and {}", feature_hash, ec.feature_space[index].values[idx]);
  }
}

void manipulate_features(
    feature_data& data, example& ec, void (*fn)(example& ec, feature_data data, int& idx_ret) = nullptr)
{
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

  if (*fn) data.manip_flag = true;  // Modify after test
  if (data.delete_flag)
    delete_feature(ec, data);
  else
    modify_feature(ec, data, idx);

  if (data.mod_flag || data.binarize_flag || data.log_flag || data.rename_flag)
    check_modify_feature(ec, data.index, data.mod_hash, idx);
}

template <bool is_learn, typename T, typename E>
void predict_or_learn(feature_data& data, T& base, E& ec)
{
  // TODO: Design a class structure
  example* copy_ec = alloc_examples(1);
  copy_example_data_with_label(copy_ec, &ec);
  data.modify = copy_ec;
  manipulate_features(data, *data.modify, modify_feature);
  if (is_learn)
  {
    if (!data.manip_flag) { base.learn(ec); }
    else
    {
      base.learn(*data.modify);
      ec.pred.scalar = std::move(data.modify->pred.scalar);
    }
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
}

void finish_example(vw& all, feature_data& data, example&) { output_and_account_example(all, *data.modify); }

VW::LEARNER::base_learner* delete_ftr_setup(setup_base_i& stack_builder)
{
  options_i& options = *stack_builder.get_options();
  vw& all = *stack_builder.get_all_pointer();
  auto data = scoped_calloc_or_throw<feature_data>();
  bool manip_ftr = false;

  option_group_definition new_options("Manipulate features");
  new_options.add(make_option("ftr_manip", manip_ftr).necessary().help("Manipulate the specified feature."));
  new_options.add(make_option("mod_ftr", data->ftr_names).help("Specify feature to be modified."));
  new_options.add(
      make_option("mod_ftr_nms", data->namespace_name).help("Specify namespace for the feature to be modified."));
  new_options.add(make_option("mod_val", data->value).help("Specify the modified value for the feature."));
  new_options.add(make_option("rename_ftr", data->mod_ftr_name).help("Specify the new name for the feature."));

  new_options.add(make_option("del_ftr", data->delete_flag)
                      .help("Option to delete the feature. No other manipulation will be applied"));
  new_options.add(
      make_option("bin_thresh", data->bin_thresh).help("Specify the threshold to binarize the feature value."));
  new_options.add(make_option("log_base", data->log_base).help("Specify the log_base for the feature."));
  if (!options.add_parse_and_check_necessary(new_options)) return nullptr;

  data->all = &all;

  if (all.options->was_supplied("mod_ftr"))
    VW::io::logger::log_warn("Setup options for modifying feature : {}", data->ftr_names);
  if (all.options->was_supplied("rename_ftr")) data->rename_flag = true;
  if (all.options->was_supplied("audit")) data->audit_flag = true;
  if (all.options->was_supplied("del_ftr"))
    data->delete_flag = true;
  else if (all.options->was_supplied("mod_val"))
    data->mod_flag = true;
  else if (all.options->was_supplied("bin_thresh"))
    data->binarize_flag = true;
  else if (all.options->was_supplied("log_base"))
    data->log_flag = true;

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
