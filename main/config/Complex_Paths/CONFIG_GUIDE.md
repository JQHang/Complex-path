## Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path_name` | string | Yes | N/A | Path name. |
| `path_schema` | list | Yes | N/A | List of tuple configurations for each hop in the path. |
| `head_node_types` | list | Yes | N/A | Head node types corresponding to each tuple configuration in the hops. |
| `head_node_indexes` | list | Yes | N/A | Head node indexes corresponding to each tuple configuration in the hops. |
| `tail_node_types` | list | Yes | N/A | Tail node types corresponding to each tuple configuration in the hops. |
| `tail_node_indexes` | list | Yes | N/A | Tail node indexes corresponding to each tuple configuration in the hops. |
| `relation_list` | list | Yes | N/A | All edge relations or path relations involved in each hop. |
| `relation_type` | list | Yes | N/A | Whether the relation is an edge or a path. |
| `edge_schema` | dict | Yes | N/A | Configuration information corresponding to the edge. |
| `head_node_columns` | list | Yes | N/A | Head nodes corresponding to the edge. |
| `edge_table_name` | string | Yes | N/A | Table name corresponding to the edge. |
| `edge_index` | int | Yes | N/A | Index corresponding to the edge. |
| `tail_node_columns` | list | Yes | N/A | Tail nodes corresponding to the edge. |
| `edge_limit` | string | No | `""` | Constraints on the attributes of the edge. |
| `edge_sample` | dict | No | `{}` | Constraints on the neighboring nodes of the edge. |
| `node_schemas` | list | No | `[]` | Configuration information for the attributes of nodes involved in the edge. |
| `path_limit` | string | No | `""` | Constraints on the attributes of all nodes and edges in the path. |