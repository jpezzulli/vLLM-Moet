# Source provenance

`Containerfile.ds4flash-0731-sm120` builds every native component from these
declared inputs:

| Component | Source | Revision or integrity value |
|---|---|---|
| Base image | Fedora 44 | `sha256:b5f1d8384c4780020a9b965ade8758b0e4a816dd2f15925666d744ba382fce1e` |
| Base vLLM | `vllm-project/vllm` | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| MoET runtime delta | `jpezzulli/vllm` | `95ef4a88c63c9ed88f2384977e05d788897af6c3` |
| Generated patch | `patch/vllm-moet-v0.24.0.patch` | `sha256:6224f37a1e0b36d2cbec6e32977f08066236444e8510b0fd085415b668f1e9f7` |
| Publication base | `jpezzulli/vLLM-Moet` | `30962258a5404df3d08fccf86b86189d04cb1456` |
| vLLM FlashAttention | `vllm-project/flash-attention` | `803020a8fa15407871341d41eba4919ade2ee1ee` |
| DeepGEMM | `deepseek-ai/DeepGEMM` | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| Python dependency lock | `requirements-runtime.lock.txt` | `sha256:f3d6db7dcb2b74d265a30566ae1fac51b0a4b702826b8537499d5a5bdf6d4dcc` |

The vLLM CMake build fetches the pinned FlashAttention revision. Its
`_vllm_fa4_cutedsl_C` component copies `flash_attn/cute/*.py` into
`vllm/vllm_flash_attn/cute` and rewrites imports to the vLLM namespace. This
generated Python subtree, FA2 and FA3 stable-ABI extensions, the base vLLM
stable-ABI extensions, and DeepGEMM are built in the image's builder stage.
The final stage receives only those declared build outputs.

The image retains the following under `/opt/provenance`:

- exact source commit files;
- generated patch and patch hash;
- pip install report and complete freeze;
- installed RPM inventory and CUDA compiler identity; and
- SHA-256 manifest for all source-built extensions, generated CuteDSL source,
  and the DeepGEMM wheel.

No file from `/opt/vllm-moet`, another live environment, a model directory, or
a host build-output directory is copied into the image.
