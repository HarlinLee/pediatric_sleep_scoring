import os
import tensorflow as tf
import tensorflow_addons as tfa

import data
import networks

def run():
    data_dir = "./nch/tfrecords/"
    checkpoint_dir = "./checkpoint"
    
    train_ds, val_ds, test_ds = data.get_sleep_stage_data(
        data_dir, batch_size=1024)
    
    model = networks.create_transformer_model(input_shape=(3840, 7), 
        num_patches=30, patch_size=128, projection_dim=64, 
        transformer_layers=8, num_heads=4, transformer_units=[128,64], 
        mlp_head_units=[256,128], num_classes=5)

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    checkpoint_filepath = os.path.join(checkpoint_dir, "model.h5")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        save_best_only=True)

    model.fit(train_ds, epochs=1, 
        validation_data=val_ds, 
        callbacks=[model_checkpoint_callback],
        class_weight={0: 0.9, 1: 5, 2: 0.9, 3: 0.9, 4: 0.9},
        verbose=2)

    model.evaluate(test_ds, verbose=2)

if __name__ == "__main__":
    run()