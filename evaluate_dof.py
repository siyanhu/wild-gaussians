import nerfbaselines


method = nerfbaselines.Method()
print(method)

# def eval_few_custom(method: WildGaussians, logger: Logger, dataset: Dataset, split: str, step: int, evaluation_protocol: EvaluationProtocol):
#     disable_tqdm = False

#     embeddings = None
#     evaluation_dataset = dataset
#     metrics = MetricsAccumulator()
#     result_no_optim = None
#     optim = None
#     optim_metrics = None
#     i = 0

#     eval_few_rows: List[List[np.ndarray]] = [[] for _ in range(len(dataset["cameras"]))]
#     if evaluation_protocol.get_name() == "nerfw":
#         # dataset = datasets.dataset_index_select(dataset, slice(None, 1))
#         optimization_dataset = horizontal_half_dataset(dataset, left=True)
#         embeddings = []
#         for i, optim in tqdm(enumerate(method.optimize_embeddings(optimization_dataset)), desc="optimizing embeddings", total=len(dataset["cameras"]), disable=disable_tqdm):
#             embeddings.append(optim["embedding"])
#             if optim_metrics is None and "metrics" in optim:
#                 optim_metrics = optim["metrics"]

#         evaluation_dataset = horizontal_half_dataset(dataset, left=False)
#         images_f = [image_to_srgb(img, dtype=np.float32) for img in evaluation_dataset["images"]]
#         for i, result_no_optim in tqdm(enumerate(method.render(evaluation_dataset["cameras"])), desc="rendering", total=len(dataset["cameras"]), disable=disable_tqdm):
#             metrics.update({
#                 k + "-nopt": v for k, v in compute_metrics(image_to_srgb(result_no_optim["color"], dtype=np.float32), images_f[i]).items()
#             })
#             eval_few_rows[i].append(image_to_srgb(result_no_optim["color"], dtype=np.uint8))
#     else:
#         images_f = [image_to_srgb(img, dtype=np.float32) for img in evaluation_dataset["images"]]

#     for i in range(len(evaluation_dataset["cameras"])):
#         eval_few_rows[i].insert(0, evaluation_dataset["images"][i])

#     result_optim = None
#     renders = []
#     for i, result_optim in tqdm(enumerate(method.render(evaluation_dataset["cameras"], embeddings=embeddings)), desc="rendering", total=len(dataset["cameras"]), disable=disable_tqdm):
#         metrics.update(compute_metrics(image_to_srgb(result_optim["color"], dtype=np.float32), images_f[i]))
#         renders.append(image_to_srgb(result_optim["color"], dtype=np.uint8))
#         eval_few_rows[i].append(image_to_srgb(result_optim["color"], dtype=np.uint8))
#     assert result_optim is not None
#     cast(Dict, evaluation_dataset)["renders"] = renders

#     with logger.add_event(step) as event:
#         for k, v in metrics.pop().items():
#             event.add_scalar(f"eval-few-{split}/{k}", v)
#         # optimization_dataset.images[i], 
#         # image_to_srgb(optim["render_output"]["color"], dtype=np.uint8),
#         if evaluation_protocol.get_name() == "nerfw":
#             assert result_no_optim is not None
#             event.add_image(f"eval-few-{split}/color", 
#                             make_image_grid(*[x for y in eval_few_rows for x in y], ncol=4),
#                             description="left: gt, middle: nopt, right: opt")
#         else:
#             event.add_image(f"eval-few-{split}/color", 
#                             make_image_grid(*[x for y in eval_few_rows for x in y], ncol=3),
#                             description="left: gt, right: render")
        
#         # Render optimization graph for PSNR, MSE
#         if optim_metrics is not None:
#             for k in ["psnr", "mse"]:
#                 metric = optim_metrics[k]
#                 event.add_plot(
#                     f"eval-few-{split}/optimization-{k}",
#                     np.stack((np.arange(len(metric)), metric), -1),
                #     axes_labels=("iteration", k),
                #     title=f"Optimization of {k} over iterations",
                # )