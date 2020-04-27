import torch
import torch.nn as nn

class CostVolumeLayer(nn.Module):

	def __init__(self):
		super(CostVolumeLayer, self).__init__()
		self.search_range = 4
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def forward(self, x1, x2):
		search_range = self.search_range

		shape = list(x1.size())
		shape[3] = (search_range * 2 + 1) ** 2
		cv = torch.zeros(shape).to(self.device)

		for i in range(-search_range, search_range + 1):
			for j in range(-search_range, search_range + 1):
				if i < 0:
					slice_h, slice_h_r = slice(None, i), slice(-i, None)
				elif i > 0:
					slice_h, slice_h_r = slice(i, None), slice(None, -i)
				else:
					slice_h, slice_h_r = slice(None), slice(None)

				if j < 0:
					slice_w, slice_w_r = slice(None, j), slice(-j, None)
				elif j > 0:
					slice_w, slice_w_r = slice(j, None), slice(None, -j)
				else:
					slice_w, slice_w_r = slice(None), slice(None)

				cv[:, slice_h, slice_w, (search_range * 2 + 1) * i + j] = (
							x1[:, slice_h, slice_w, :] * x2[:, slice_h_r, slice_w_r, :]).sum(3)

		return cv / shape[1]