#!/usr/bin/env python3

import json

def main():
    config_path = "/Users/z/work/zen/zen-nano/model/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["architectures"] = ["Qwen3ForCausalLM"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

if __name__ == "__main__":
    main()
