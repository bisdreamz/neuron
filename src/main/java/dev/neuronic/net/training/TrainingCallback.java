package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;

/**
 * Interface for training callbacks that can monitor and control the training process.
 * 
 * Callbacks provide hooks at various points during training:
 * - Training start/end
 * - Epoch start/end
 * - Batch end (optional)
 * 
 * Use cases:
 * - Progress monitoring and logging
 * - Early stopping based on metrics
 * - Learning rate scheduling
 * - Model checkpointing
 * - Visualization and plotting
 */
public interface TrainingCallback {
    
    /**
     * Called at the beginning of training.
     * 
     * @param model the neural network being trained
     * @param metrics metrics collector that will track training progress
     */
    default void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {}
    
    /**
     * Called at the end of each epoch.
     * 
     * @param epoch current epoch number (0-based)
     * @param metrics current training metrics
     */
    default void onEpochEnd(int epoch, TrainingMetrics metrics) {}
    
    /**
     * Called at the end of training.
     * 
     * @param model the trained neural network
     * @param metrics final training metrics
     */
    default void onTrainingEnd(NeuralNet model, TrainingMetrics metrics) {}
}