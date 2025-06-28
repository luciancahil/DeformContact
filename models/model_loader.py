from models.model import GraphNet, GraphNetTwo

def load_model(config, multiplier, graphNum = 1):
    if(graphNum == 1):

        model = GraphNet(input_dims=config.network.input_dims,
                        hidden_dim=config.network.hidden_dim * multiplier,
                        output_dim=config.network.output_dim,
                        encoder_layers=config.network.encoder_layers,
                        decoder_layers=config.network.decoder_layers,
                        dropout_rate=config.network.dropout_rate,
                        knn_k=config.network.knn_k,
                        use_mha= config.network.use_mha,
                        num_mha_heads= config.network.num_mha_heads,
                        backbone=config.network.backbone,
                        mode=config.network.mode,
                        edge_dim=config.network.edge_dim)
    else:
        model = GraphNetTwo(input_dims=config.network.input_dims,
                hidden_dim=config.network.hidden_dim * multiplier,
                output_dim=config.network.output_dim,
                encoder_layers=config.network.encoder_layers,
                decoder_layers=config.network.decoder_layers,
                dropout_rate=config.network.dropout_rate,
                knn_k=config.network.knn_k,
                use_mha= config.network.use_mha,
                num_mha_heads= config.network.num_mha_heads,
                backbone=config.network.backbone,
                mode=config.network.mode,
                edge_dim=config.network.edge_dim)

    return model

    
