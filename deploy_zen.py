#!/usr/bin/env python3
"""
Unified Zen Deployment System
Deploy any Zen family model with optimal configuration
"""

import os
import sys
import json
import yaml
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict
import torch
import psutil
import fire
from enum import Enum
import platform
import docker
import kubernetes
from kubernetes import client, config

class ZenModel(Enum):
    """Available Zen family models"""
    OMNI = "zen-omni"
    CODER = "zen-coder"
    NANO = "zen-nano"
    NEXT = "zen-next"

class DeploymentTarget(Enum):
    """Deployment targets"""
    LOCAL = "local"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    HANZO_CLOUD = "hanzo"
    EDGE = "edge"
    MOBILE = "mobile"
    BROWSER = "browser"

@dataclass
class DeploymentConfig:
    """Unified deployment configuration"""
    model: ZenModel
    target: DeploymentTarget
    
    # Model configuration
    model_path: Optional[str] = None
    quantization: str = "auto"  # auto, 1bit, 2bit, 4bit, int8, fp16
    progressive_download: bool = True
    initial_quality: float = 0.72  # For PD-LLM
    
    # Resource configuration
    max_memory: Optional[str] = None  # e.g., "8GB"
    max_cpu: Optional[int] = None
    gpu_device: Optional[str] = None  # cuda:0, mps, cpu
    
    # Service configuration
    port: int = 8080
    host: str = "0.0.0.0"
    workers: int = 1
    max_batch_size: int = 32
    max_sequence_length: int = 2048
    
    # Performance configuration
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    use_bitdelta: bool = True
    cache_size: str = "1GB"
    
    # Deployment specifics
    replicas: int = 1
    autoscale: bool = False
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: int = 80
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    
    # Security
    enable_auth: bool = False
    api_key: Optional[str] = None
    tls_cert: Optional[str] = None
    tls_key: Optional[str] = None

class SystemAnalyzer:
    """Analyze system capabilities for optimal deployment"""
    
    @staticmethod
    def analyze() -> Dict:
        """Analyze current system"""
        analysis = {
            "cpu": {
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
                "usage": psutil.cpu_percent(interval=1)
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "usage_percent": psutil.virtual_memory().percent
            },
            "gpu": SystemAnalyzer._check_gpu(),
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": sys.version.split()[0]
            },
            "network": {
                "bandwidth_estimate": SystemAnalyzer._estimate_bandwidth()
            }
        }
        
        return analysis
    
    @staticmethod
    def _check_gpu() -> Dict:
        """Check GPU availability"""
        gpu_info = {"available": False, "type": "none"}
        
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["type"] = "cuda"
            gpu_info["count"] = torch.cuda.device_count()
            gpu_info["names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            gpu_info["memory_gb"] = [torch.cuda.get_device_properties(i).total_memory / (1024**3) 
                                     for i in range(torch.cuda.device_count())]
        elif torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["type"] = "mps"
            gpu_info["device"] = "Apple Silicon"
        
        return gpu_info
    
    @staticmethod
    def _estimate_bandwidth() -> float:
        """Estimate network bandwidth (Mbps)"""
        # Simple estimation, in production use speedtest
        return 100.0  # Default 100 Mbps

class DeploymentOptimizer:
    """Optimize deployment configuration based on system and requirements"""
    
    def __init__(self, config: DeploymentConfig, system_info: Dict):
        self.config = config
        self.system_info = system_info
    
    def optimize(self) -> DeploymentConfig:
        """Optimize configuration for system"""
        optimized = self.config
        
        # Auto-select quantization based on available memory
        if optimized.quantization == "auto":
            optimized.quantization = self._select_quantization()
        
        # Auto-configure resources
        if not optimized.max_memory:
            optimized.max_memory = self._calculate_max_memory()
        
        if not optimized.max_cpu:
            optimized.max_cpu = self._calculate_max_cpu()
        
        if not optimized.gpu_device:
            optimized.gpu_device = self._select_gpu_device()
        
        # Optimize for model type
        optimized = self._optimize_for_model(optimized)
        
        # Optimize for deployment target
        optimized = self._optimize_for_target(optimized)
        
        return optimized
    
    def _select_quantization(self) -> str:
        """Select optimal quantization based on resources"""
        available_memory = self.system_info["memory"]["available_gb"]
        
        if self.config.model == ZenModel.NANO:
            if available_memory < 1:
                return "1bit"
            elif available_memory < 2:
                return "2bit"
            elif available_memory < 4:
                return "4bit"
            else:
                return "int8"
        
        elif self.config.model in [ZenModel.OMNI, ZenModel.CODER]:
            if available_memory < 8:
                return "4bit"
            elif available_memory < 16:
                return "int8"
            else:
                return "fp16"
        
        return "fp16"
    
    def _calculate_max_memory(self) -> str:
        """Calculate maximum memory allocation"""
        available = self.system_info["memory"]["available_gb"]
        
        # Reserve memory for system
        if self.config.target == DeploymentTarget.LOCAL:
            usable = available * 0.8
        else:
            usable = available * 0.9
        
        return f"{int(usable)}GB"
    
    def _calculate_max_cpu(self) -> int:
        """Calculate maximum CPU allocation"""
        cores = self.system_info["cpu"]["cores"]
        
        if self.config.target == DeploymentTarget.LOCAL:
            return max(1, cores - 1)
        else:
            return cores
    
    def _select_gpu_device(self) -> str:
        """Select optimal GPU device"""
        gpu_info = self.system_info["gpu"]
        
        if not gpu_info["available"]:
            return "cpu"
        
        if gpu_info["type"] == "cuda":
            # Select GPU with most memory
            if "memory_gb" in gpu_info:
                best_gpu = max(enumerate(gpu_info["memory_gb"]), key=lambda x: x[1])[0]
                return f"cuda:{best_gpu}"
            return "cuda:0"
        
        elif gpu_info["type"] == "mps":
            return "mps"
        
        return "cpu"
    
    def _optimize_for_model(self, config: DeploymentConfig) -> DeploymentConfig:
        """Model-specific optimizations"""
        if config.model == ZenModel.NANO:
            config.use_flash_attention = False  # Not needed for small model
            config.max_batch_size = 64  # Can handle larger batches
            config.workers = 1  # Single worker sufficient
        
        elif config.model == ZenModel.CODER:
            config.max_sequence_length = 4096  # Longer context for code
            config.use_kv_cache = True  # Important for code generation
        
        elif config.model == ZenModel.OMNI:
            config.progressive_download = True  # Always use PD-LLM
            config.use_bitdelta = True  # Enable personalization
        
        return config
    
    def _optimize_for_target(self, config: DeploymentConfig) -> DeploymentConfig:
        """Target-specific optimizations"""
        if config.target == DeploymentTarget.EDGE:
            config.quantization = "2bit"
            config.progressive_download = True
            config.workers = 1
        
        elif config.target == DeploymentTarget.MOBILE:
            config.quantization = "1bit"
            config.max_batch_size = 1
            config.workers = 1
        
        elif config.target == DeploymentTarget.KUBERNETES:
            config.autoscale = True
            config.enable_monitoring = True
        
        return config

class LocalDeployer:
    """Deploy Zen models locally"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    async def deploy(self) -> Dict:
        """Deploy model locally"""
        print(f"Deploying {self.config.model.value} locally...")
        
        # Create service script
        service_script = self._generate_service_script()
        
        # Save script
        script_path = Path(f"/tmp/zen_{self.config.model.value}_service.py")
        with open(script_path, 'w') as f:
            f.write(service_script)
        
        # Start service
        cmd = [
            sys.executable,
            str(script_path),
            "--port", str(self.config.port),
            "--host", self.config.host,
            "--workers", str(self.config.workers)
        ]
        
        if self.config.gpu_device:
            env = os.environ.copy()
            if "cuda" in self.config.gpu_device:
                env["CUDA_VISIBLE_DEVICES"] = self.config.gpu_device.split(":")[-1]
        else:
            env = os.environ.copy()
        
        process = subprocess.Popen(cmd, env=env)
        
        # Wait for service to start
        await asyncio.sleep(5)
        
        return {
            "status": "running",
            "pid": process.pid,
            "url": f"http://{self.config.host}:{self.config.port}",
            "model": self.config.model.value,
            "config": asdict(self.config)
        }
    
    def _generate_service_script(self) -> str:
        """Generate service script"""
        return f"""#!/usr/bin/env python3
import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

app = FastAPI()

# Load model
model_path = "{self.config.model_path or self.config.model.value}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16 if "{self.config.quantization}" == "fp16" else torch.float32
)

@app.post("/generate")
async def generate(prompt: str, max_length: int = 100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    return {{"response": tokenizer.decode(outputs[0])}}

@app.get("/health")
async def health():
    return {{"status": "healthy", "model": "{self.config.model.value}"}}

if __name__ == "__main__":
    uvicorn.run(app, host="{self.config.host}", port={self.config.port}, workers={self.config.workers})
"""

class DockerDeployer:
    """Deploy Zen models in Docker"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = docker.from_env()
    
    async def deploy(self) -> Dict:
        """Deploy model in Docker container"""
        print(f"Deploying {self.config.model.value} in Docker...")
        
        # Build Docker image
        dockerfile = self._generate_dockerfile()
        image_tag = f"zen-{self.config.model.value}:latest"
        
        # Build image
        image_path = Path(f"/tmp/zen_{self.config.model.value}_docker")
        image_path.mkdir(exist_ok=True)
        
        with open(image_path / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        
        print("Building Docker image...")
        image, logs = self.client.images.build(
            path=str(image_path),
            tag=image_tag,
            rm=True
        )
        
        # Run container
        container = self.client.containers.run(
            image_tag,
            detach=True,
            ports={f"{self.config.port}/tcp": self.config.port},
            environment={
                "MODEL": self.config.model.value,
                "QUANTIZATION": self.config.quantization,
                "MAX_MEMORY": self.config.max_memory
            },
            runtime="nvidia" if "cuda" in self.config.gpu_device else None
        )
        
        return {
            "status": "running",
            "container_id": container.id,
            "url": f"http://localhost:{self.config.port}",
            "model": self.config.model.value,
            "image": image_tag
        }
    
    def _generate_dockerfile(self) -> str:
        """Generate Dockerfile"""
        return f"""FROM python:3.10-slim

WORKDIR /app

RUN pip install torch transformers fastapi uvicorn

COPY service.py /app/

ENV MODEL={self.config.model.value}
ENV PORT={self.config.port}

CMD ["python", "service.py"]
"""

class KubernetesDeployer:
    """Deploy Zen models on Kubernetes"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
        # Load k8s config
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        
        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
    
    async def deploy(self) -> Dict:
        """Deploy model on Kubernetes"""
        print(f"Deploying {self.config.model.value} on Kubernetes...")
        
        # Create deployment
        deployment = self._create_deployment()
        self.apps_v1.create_namespaced_deployment(
            namespace="default",
            body=deployment
        )
        
        # Create service
        service = self._create_service()
        self.v1.create_namespaced_service(
            namespace="default",
            body=service
        )
        
        # Create HPA if autoscaling
        if self.config.autoscale:
            hpa = self._create_hpa()
            # Create HPA using autoscaling API
        
        return {
            "status": "deployed",
            "deployment": f"zen-{self.config.model.value}",
            "service": f"zen-{self.config.model.value}-service",
            "replicas": self.config.replicas,
            "autoscale": self.config.autoscale
        }
    
    def _create_deployment(self) -> Dict:
        """Create Kubernetes deployment spec"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"zen-{self.config.model.value}"
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": f"zen-{self.config.model.value}"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": f"zen-{self.config.model.value}"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "zen-model",
                            "image": f"hanzo-ai/zen-{self.config.model.value}:latest",
                            "ports": [{
                                "containerPort": self.config.port
                            }],
                            "resources": {
                                "requests": {
                                    "memory": self.config.max_memory,
                                    "cpu": str(self.config.max_cpu)
                                },
                                "limits": {
                                    "memory": self.config.max_memory,
                                    "nvidia.com/gpu": "1" if "cuda" in self.config.gpu_device else "0"
                                }
                            }
                        }]
                    }
                }
            }
        }
    
    def _create_service(self) -> Dict:
        """Create Kubernetes service spec"""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"zen-{self.config.model.value}-service"
            },
            "spec": {
                "selector": {
                    "app": f"zen-{self.config.model.value}"
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": self.config.port,
                    "targetPort": self.config.port
                }],
                "type": "LoadBalancer"
            }
        }
    
    def _create_hpa(self) -> Dict:
        """Create Horizontal Pod Autoscaler spec"""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"zen-{self.config.model.value}-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"zen-{self.config.model.value}"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [{
                    "type": "Resource",
                    "resource": {
                        "name": "cpu",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": self.config.target_cpu_percent
                        }
                    }
                }]
            }
        }

class ZenDeployer:
    """Main Zen deployment orchestrator"""
    
    def __init__(self):
        self.deployers = {
            DeploymentTarget.LOCAL: LocalDeployer,
            DeploymentTarget.DOCKER: DockerDeployer,
            DeploymentTarget.KUBERNETES: KubernetesDeployer
        }
    
    async def deploy(self, config: DeploymentConfig) -> Dict:
        """Deploy Zen model with optimal configuration"""
        
        # Analyze system
        print("Analyzing system capabilities...")
        system_info = SystemAnalyzer.analyze()
        self._print_system_info(system_info)
        
        # Optimize configuration
        print("\nOptimizing deployment configuration...")
        optimizer = DeploymentOptimizer(config, system_info)
        optimized_config = optimizer.optimize()
        self._print_config(optimized_config)
        
        # Deploy
        print(f"\nDeploying to {config.target.value}...")
        deployer_class = self.deployers.get(config.target, LocalDeployer)
        deployer = deployer_class(optimized_config)
        
        result = await deployer.deploy()
        
        # Print results
        self._print_deployment_result(result)
        
        return result
    
    def _print_system_info(self, info: Dict):
        """Print system information"""
        print(f"  CPU: {info['cpu']['cores']} cores, {info['cpu']['threads']} threads")
        print(f"  Memory: {info['memory']['available_gb']:.1f}/{info['memory']['total_gb']:.1f} GB available")
        print(f"  GPU: {info['gpu']['type']} - {info['gpu'].get('names', ['None'])[0] if info['gpu']['available'] else 'Not available'}")
        print(f"  Platform: {info['platform']['system']} {info['platform']['machine']}")
    
    def _print_config(self, config: DeploymentConfig):
        """Print optimized configuration"""
        print(f"  Model: {config.model.value}")
        print(f"  Quantization: {config.quantization}")
        print(f"  Memory: {config.max_memory}")
        print(f"  GPU: {config.gpu_device}")
        print(f"  Progressive Download: {config.progressive_download}")
    
    def _print_deployment_result(self, result: Dict):
        """Print deployment result"""
        print(f"\n✅ Deployment successful!")
        print(f"  Status: {result.get('status')}")
        print(f"  URL: {result.get('url', 'N/A')}")
        print(f"  Model: {result.get('model')}")

async def main(
    model: str = "zen-omni",
    target: str = "local",
    quantization: str = "auto",
    progressive: bool = True,
    port: int = 8080,
    **kwargs
):
    """
    Deploy Zen family models
    
    Args:
        model: Model to deploy (zen-omni, zen-coder, zen-nano, zen-next)
        target: Deployment target (local, docker, kubernetes, hanzo, edge, mobile)
        quantization: Quantization mode (auto, 1bit, 2bit, 4bit, int8, fp16)
        progressive: Enable progressive download
        port: Service port
    """
    
    # Create configuration
    config = DeploymentConfig(
        model=ZenModel(model),
        target=DeploymentTarget(target),
        quantization=quantization,
        progressive_download=progressive,
        port=port,
        **kwargs
    )
    
    # Deploy
    deployer = ZenDeployer()
    await deployer.deploy(config)

if __name__ == "__main__":
    # Run with fire for CLI
    fire.Fire(lambda **kwargs: asyncio.run(main(**kwargs)))