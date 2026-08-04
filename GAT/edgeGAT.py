'''GAT Layer'''

# GAT的PyG实现
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax as pyg_softmax
from torch.distributions import Categorical

'''GAT节点与边参数设置'''

# 节点类型定义
NULL_NODE = 0
EGO_NODE = 1
PRO_NODE = 2
ALI_NODE = 3

# 边类型定义
AGENT_PROPOSAL = 0
AGENT_ALIGN = 1

# 扇区风险评分,后续需要与guidance对齐
SECTOR_RISK_SCORE = 17

# 节点特征维度定义
NODE_FEATURE_DIMS = {
    NULL_NODE: 3,
    EGO_NODE: 5+SECTOR_RISK_SCORE,
    PRO_NODE: 7,
    ALI_NODE: 5
}

# 边特征维度定义
EDGE_FEATURE_DIMS = {
    AGENT_PROPOSAL: 1, 
    AGENT_ALIGN: 2
}

def pad_features(values, target_dim):
    '''Pad features to a target dimension using zeros'''
    return values + [0.0] * (target_dim - len(values))
def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.ReLU(),
        nn.Linear(output_dim, output_dim),
    )

class RiskAwareGATSelector(nn.Module):
    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout=0.0
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError('hidden_dim must be divisible by num_heads')

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 保存节点和边特征维度配置，避免后续投影过程依赖可变的全局字典
        self.NODE_FEATURE_DIMS = dict(NODE_FEATURE_DIMS)
        self.EDGE_FEATURE_DIMS = dict(EDGE_FEATURE_DIMS)

        # 按照节点类型构建独立encoder，并统一投影到hidden_dim维隐空间
        self.node_encoders = nn.ModuleDict({
            str(node_type): build_mlp(input_dim, hidden_dim)
            for node_type, input_dim in self.NODE_FEATURE_DIMS.items()
        })

        # 按照边类型构建独立encoder，使不同语义边获得统一维度表示
        self.edge_encoders = nn.ModuleDict({
            str(edge_type): build_mlp(input_dim, hidden_dim)
            for edge_type, input_dim in self.EDGE_FEATURE_DIMS.items()
        })

        # 第一层Edge-Enhanced GAT，将编码后的边特征纳入注意力计算
        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim,
            add_self_loops=True,
            fill_value=0.0,
            dropout=dropout,
        )

        # 第二层Edge-Enhanced GAT，进一步聚合候选点的局部风险信息
        self.gat2 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=hidden_dim,
            add_self_loops=True,
            fill_value=0.0,
            dropout=dropout,
        )

        # 对null节点和候选路径点逐节点输出未归一化logit
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )


    def project_by_type(
        self,
        raw_feature,
        type_index,
        feature_dims,
        encoders,
        output_dim,
    ):
        '''按照节点或边类型，将不同维度的原始特征投影到统一隐空间'''

        # 类型投影要求二维特征矩阵与一维类型索引逐行对应
        if raw_feature.ndim != 2:
            raise ValueError('raw_feature must be a two-dimensional tensor')
        if type_index.ndim != 1:
            raise ValueError('type_index must be a one-dimensional tensor')
        if raw_feature.size(0) != type_index.numel():
            raise ValueError('raw_feature and type_index must have the same length')
        if not raw_feature.is_floating_point():
            raise TypeError('raw_feature must use a floating-point dtype')

        # 主动检查未注册类型，避免对应节点或边被静默编码为零向量
        observed_types = set(type_index.detach().cpu().unique().tolist())
        unknown_types = observed_types - set(feature_dims.keys())
        if unknown_types:
            raise ValueError(f'found unregistered feature types: {sorted(unknown_types)}')

        # 投影结果继承输入张量的设备和数据类型，兼容混合精度训练
        encoded = raw_feature.new_zeros((raw_feature.size(0), output_dim))

        for type_id, valid_dim in feature_dims.items():
            if valid_dim <= 0 or valid_dim > raw_feature.size(1):
                raise ValueError(
                    f'invalid feature dimension {valid_dim} for type {type_id}'
                )

            # 获取当前类型对应的节点或边索引
            indices = (
                type_index == type_id
            ).nonzero(as_tuple=False).flatten()

            if indices.numel() == 0:
                continue  # 当前图不存在该类型时跳过对应encoder

            # 各类型有效特征位于填充特征矩阵的前valid_dim列
            current_feature = raw_feature[indices, :valid_dim]
            current_encoded = encoders[str(type_id)](current_feature)

            # 按照原始索引写回统一特征矩阵，并保持梯度传播链路
            encoded = encoded.index_copy(0, indices, current_encoded)

        return encoded

    def forward(self, data):
        # 检查GAT计算所需的图属性是否完整
        required_attributes = (
            'x', 'node_type', 'edge_index', 'edge_attr', 'edge_type', 'selectable'
        )
        missing_attributes = [
            name for name in required_attributes
            if getattr(data, name, None) is None
        ]
        if missing_attributes:
            raise ValueError(f'missing graph attributes: {missing_attributes}')

        if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
            raise ValueError('edge_index must have shape [2, num_edges]')
        if data.edge_attr.size(0) != data.edge_index.size(1):
            raise ValueError('edge_attr and edge_index must describe the same edges')

        # 按照节点类型编码异构节点特征
        node_embedding = self.project_by_type(
            data.x,
            data.node_type,
            self.NODE_FEATURE_DIMS,
            self.node_encoders,
            self.hidden_dim,
        )

        # 按照边类型编码关系特征，并与原始边顺序保持一致
        edge_embedding = self.project_by_type(
            data.edge_attr,
            data.edge_type,
            self.EDGE_FEATURE_DIMS,
            self.edge_encoders,
            self.hidden_dim,
        )

        # 第一层EGAT同时返回注意力权重，用于后续风险关系解释
        node_embedding, attention_info = self.gat1(
            x=node_embedding,
            edge_index=data.edge_index,
            edge_attr=edge_embedding,
            return_attention_weights=True,
        )
        attention_edge_index, attention_weight = attention_info
        node_embedding = F.elu(node_embedding)

        # 第二层EGAT进一步更新候选点的风险感知表示
        node_embedding = self.gat2(
            x=node_embedding,
            edge_index=data.edge_index,
            edge_attr=edge_embedding,
        )
        node_embedding = F.elu(node_embedding)

        # 将布尔候选掩码或候选索引统一转换为原始图节点索引
        if data.selectable.ndim != 1:
            raise ValueError('selectable must be a one-dimensional tensor')
        if data.selectable.dtype == torch.bool:
            if data.selectable.numel() != node_embedding.size(0):
                raise ValueError('boolean selectable mask must match the number of nodes')
            selectable_index = data.selectable.nonzero(as_tuple=False).flatten()
        else:
            selectable_index = data.selectable.to(
                device=node_embedding.device, dtype=torch.long
            )
            if (selectable_index < 0).any() or (
                selectable_index >= node_embedding.size(0)
            ).any():
                raise IndexError('selectable contains an out-of-range node index')

        if selectable_index.numel() == 0:
            raise ValueError('each agent graph must contain at least one selectable node')

        # 仅对null节点和候选路径点进行节点级打分
        selectable_embedding = node_embedding[selectable_index]
        scores = self.score_mlp(selectable_embedding).squeeze(-1)

        # 单图在候选集合内归一化，批图则按照智能体局部图分别归一化
        selectable_batch = None
        batch_index = getattr(data, 'batch', None)
        if batch_index is None:
            probs = F.softmax(scores, dim=0)
        else:
            if batch_index.ndim != 1 or batch_index.numel() != node_embedding.size(0):
                raise ValueError('batch must provide one graph index for each node')
            selectable_batch = batch_index[selectable_index]
            probs = pyg_softmax(scores, index=selectable_batch)

        return {
            'node_embedding': node_embedding,
            'attention_edge_index': attention_edge_index,
            'attention_weight': attention_weight,
            'selectable_index': selectable_index,
            'selectable_batch': selectable_batch,
            'scores': scores,
            'probs': probs,
        }

class GraphBuilder:
    '''
    异构图构建器，将异构图数据转换为异构图表示。
    '''
    def __init__(
        self,
        node_feature_dims: dict[int, int],
        edge_feature_dims: dict[int, int],
        device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        

def SceneGraphReader(file_path, node_feature_dims, edge_feature_dims):
    '''读取外部场景信息，组织为能够由异构图读取的图结构形式'''
