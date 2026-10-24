"""Lightning wrapper for stage-2 PLM + K-token compression captioning (no Q-Former)."""
from model.blip2_opt_plm8 import Blip2OPT_PLM8
from model.blip2_stage2_plm import Blip2Stage2PLM


class Blip2Stage2PLM8(Blip2Stage2PLM):
    def _build_blip2(self, args):
        return Blip2OPT_PLM8(
            args.plm_model,
            args.plm_tune,
            args.llm_name,
            args.llm_tune,
            args.peft_dir,
            getattr(args, "num_prefix_tokens", 8),
            args,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Blip2Stage2PLM.add_model_specific_args(parent_parser)
        group = parser.add_argument_group("ProtBlip2PLM8")
        group.add_argument(
            "--num_prefix_tokens",
            type=int,
            default=8,
            help="compress PLM hidden states [L,H_llm] to this many Galactica prefix tokens",
        )
        return parser
