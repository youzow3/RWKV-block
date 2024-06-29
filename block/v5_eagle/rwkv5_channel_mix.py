import torch
from torch import nn
from .rwkv5_block_config_map import RWKV5BlockConfigMap, RWKV5BlockConfigMapNormalizer

class RWKV5ChannelMix(torch.nn.Module):
	'''
	ChannelMix block for RWKV
	This is similar to transformer FFN block
	'''

	def __init__(self, configMap: RWKV5BlockConfigMap|any):
		super().__init__()

		cMap:RWKV5BlockConfigMap = RWKV5BlockConfigMapNormalizer(configMap)
		self.configMap = cMap

		# Get required props
		n_embed = cMap.n_embed
		n_layer = cMap.n_layer

		# Get optional props
		dim_ffn = cMap.get_dim_ffn()
		layer_id = cMap.get_layer_id(0)
		device = cMap.get_device('cpu')
		dtype = cMap.get_dtype('float')

		# Build the various params
		# ---
		with torch.no_grad():  # fancy init of time_mix
			ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
			ddd = torch.ones(1, 1, n_embed)
			for i in range(n_embed):
				ddd[0, 0, i] = i / n_embed
			self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0).to(device, dtype=dtype))
			self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0).to(device, dtype=dtype))

		self.key = nn.Linear(n_embed, dim_ffn, bias=False, device=device, dtype=dtype)
		self.receptance = nn.Linear(n_embed, n_embed, bias=False, device=device, dtype=dtype)
		self.value = nn.Linear(dim_ffn, n_embed, bias=False, device=device, dtype=dtype)

	def forward(self, x: torch.Tensor, last_state: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
		'''
		Forwarding channel mix given the input tokens and states.
		
		Given:
		- Incoming token embedding size of shape [batch_size, seq_len, embedding_size]
		- Incoming channel mix, shift states of the various batches [batch_size, state_size]
		
		Returns a pair 
		- Output embedding of shape [batch_size, seq_len, embedding_size]
		- Output channel mix, shift state of shape [batch_size, state_size]
		'''
		# last_state = last_state.to(self.key.weight.device)

		# 7B 4090
		# - no compile:      0.3243246078491211 ms
		# - default compile: 0.3113429546356201 ms
		# - max auto tune:   0.30689454078674316 ms
		# ---
		xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]),
						  dim=1)
		xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
		xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
		kv = self.value( torch.relu( self.key(xk) ) ** 2 )
		return (torch.sigmoid(self.receptance(xr)) * kv,
				(x[:, -1]))

		# 7B 4090 
		# - no compile:      0.3181338310241699 ms
		# - default compile: 0.3143901824951172 ms
		# - max auto tune:   0.3115804195404053 ms
		# ---
		# xx = torch.concat((last_state.unsqueeze(1), x[:, :-1]), dim=1)
		# xk,xr = torch.lerp(xx, x, torch.stack((self.time_mix_k,self.time_mix_r)))
		# kv = self.value( torch.relu( self.key(xk) ) ** 2 )
		# return torch.sigmoid(self.receptance(xr)) * kv, x[:,-1]
