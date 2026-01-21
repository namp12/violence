"""
3D CNN model for violence detection.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import List, Tuple


def create_conv3d_block(filters: int, kernel_size: Tuple[int, int, int] = (3, 3, 3),
                        pool_size: Tuple[int, int, int] = (2, 2, 2),
                        activation: str = 'relu'):
    """
    Create a 3D convolutional block.
    
    Args:
        filters: Number of filters
        kernel_size: Kernel size for Conv3D
        pool_size: Pool size for MaxPooling3D
        activation: Activation function
        
    Returns:
        Sequential model block
    """
    block = keras.Sequential([
        layers.Conv3D(filters, kernel_size, activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.Conv3D(filters, kernel_size, activation=activation, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=pool_size)
    ])
    return block


def create_model(input_shape: Tuple[int, int, int, int] = (16, 112, 112, 3),
                num_classes: int = 2,
                conv_blocks: List[dict] = None,
                dense_layers: List[dict] = None) -> keras.Model:
    """
    Create 3D CNN model for violence detection.
    
    Args:
        input_shape: Input shape (num_frames, height, width, channels)
        num_classes: Number of output classes
        conv_blocks: List of convolutional block configurations
        dense_layers: List of dense layer configurations
        
    Returns:
        Keras model
    """
    # Default configuration if not provided
    if conv_blocks is None:
        conv_blocks = [
            {'filters': 32, 'kernel_size': (3, 3, 3), 'pool_size': (2, 2, 2)},
            {'filters': 64, 'kernel_size': (3, 3, 3), 'pool_size': (2, 2, 2)},
            {'filters': 128, 'kernel_size': (3, 3, 3), 'pool_size': (2, 2, 2)},
            {'filters': 256, 'kernel_size': (3, 3, 3), 'pool_size': (2, 2, 2)}
        ]
    
    if dense_layers is None:
        dense_layers = [
            {'units': 512, 'dropout': 0.5, 'batch_norm': True},
            {'units': 256, 'dropout': 0.5, 'batch_norm': False}
        ]
    
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional blocks
    x = inputs
    for i, block_config in enumerate(conv_blocks):
        x = create_conv3d_block(
            filters=block_config['filters'],
            kernel_size=block_config.get('kernel_size', (3, 3, 3)),
            pool_size=block_config.get('pool_size', (2, 2, 2)),
            activation=block_config.get('activation', 'relu')
        )(x)
    
    # Flatten
    x = layers.Flatten()(x)
    
    # Dense layers
    for dense_config in dense_layers:
        x = layers.Dense(dense_config['units'], activation='relu')(x)
        
        if dense_config.get('batch_norm', False):
            x = layers.BatchNormalization()(x)
        
        if dense_config.get('dropout', 0) > 0:
            x = layers.Dropout(dense_config['dropout'])(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='CNN3D_Violence_Detection')
    
    return model


def compile_model(model: keras.Model, learning_rate: float = 0.0001,
                 optimizer: str = 'adam', loss: str = 'categorical_crossentropy',
                 metrics: List[str] = None) -> keras.Model:
    """
    Compile the model.
    
    Args:
        model: Keras model
        learning_rate: Learning rate
        optimizer: Optimizer name
        loss: Loss function
        metrics: List of metrics
        
    Returns:
        Compiled model
    """
    if metrics is None:
        metrics = ['accuracy']
    
    # Create optimizer
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = optimizer
    
    # Compile
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=metrics
    )
    
    return model


def create_and_compile_model(config: dict = None) -> keras.Model:
    """
    Create and compile model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    if config is None:
        # Use default configuration
        model = create_model()
        model = compile_model(model)
    else:
        # Extract model config
        model_config = config.get('model', {})
        training_config = config.get('training', {})
        
        # Create model
        model = create_model(
            input_shape=tuple(model_config.get('input_shape', [16, 112, 112, 3])),
            num_classes=model_config.get('num_classes', 2),
            conv_blocks=model_config.get('conv_blocks'),
            dense_layers=model_config.get('dense_layers')
        )
        
        # Compile model
        model = compile_model(
            model,
            learning_rate=training_config.get('initial_learning_rate', 0.0001),
            optimizer=training_config.get('optimizer', 'adam'),
            loss=training_config.get('loss', 'categorical_crossentropy'),
            metrics=training_config.get('metrics', ['accuracy'])
        )
    
    return model


if __name__ == '__main__':
    # Test model creation
    print("Creating 3D CNN model...")
    model = create_model()
    model = compile_model(model)
    
    print("\nModel Summary:")
    model.summary()
    
    print(f"\nTotal parameters: {model.count_params():,}")
