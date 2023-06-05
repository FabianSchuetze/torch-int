#ifndef BMM_H
#define BMM_H
#include <torch/types.h>
torch::Tensor bmm_s8t_s8n_f32t(torch::Tensor A, torch::Tensor B, float alpha);

torch::Tensor bmm_s8t_s8n_s8t(torch::Tensor A, torch::Tensor B, float alpha);

torch::Tensor bmm_s8t_s8n_s32t(torch::Tensor A, torch::Tensor B);

torch::Tensor bmm_f32t_f32n_f32t(torch::Tensor A, torch::Tensor B, float alpha);

#endif // BMM_H
