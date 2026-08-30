# Table 1 baseline imports

The following upstream implementations are vendored under this directory for
reproducible baseline integration:

| Method | Upstream source | Local code | Runner status |
|---|---|---|---|
| AirNet | `XLearning-SCU/2022-CVPR-AirNet` | `airnet_net/`, `airnet_utils/` | blocked by the upstream `mmcv.ops` dependency |
| TransWeather | `jeya-maria-jose/TransWeather` | `transweather_model.py`, `base_networks.py` | requires the upstream `timm/mmcv` stack and a runner adapter |
| DFPIR | `TxpHome/DFPIR` | `dfpir_net/` | requires the upstream CLIP/degradation-conditioning path and a runner adapter |
| R2R | `cscxwang/R2R` | `r2r_net/`, `r2r_options_5D.py` | requires its degradation-memory training protocol and a runner adapter |

These methods must not be silently replaced by another architecture.  Until
their upstream dependencies and input protocols are adapted, the formal
comparison runner uses the already validated Restormer, PromptIR, PReNet, GRL,
and SiblingRestore entries.  MFNET is not vendored because the v2.0 protocol
does not identify a public official implementation.
