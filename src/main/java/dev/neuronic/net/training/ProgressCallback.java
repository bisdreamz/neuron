package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.training.TrainingMetrics.EpochMetrics;

/**
 * Progress reporting callback that prints training progress to console.
 * 
 * Supports two modes:
 * - Simple: Shows epoch progress with key metrics
 * - Detailed: Shows batch-level progress and additional statistics
 */
public class ProgressCallback implements TrainingCallback {
    
    private final boolean detailed;
    private final boolean languageModelMode;
    private long trainingStartTime;
    private int totalEpochs;
    
    public ProgressCallback(boolean detailed) {
        this.detailed = detailed;
        this.languageModelMode = false;
        this.totalEpochs = -1; // Unknown by default
    }
    
    public ProgressCallback(boolean detailed, int totalEpochs) {
        this.detailed = detailed;
        this.languageModelMode = false;
        this.totalEpochs = totalEpochs;
    }
    
    public ProgressCallback(boolean detailed, boolean languageModelMode) {
        this.detailed = detailed;
        this.languageModelMode = languageModelMode;
        this.totalEpochs = -1;
    }
    
    public ProgressCallback(boolean detailed, boolean languageModelMode, int totalEpochs) {
        this.detailed = detailed;
        this.languageModelMode = languageModelMode;
        this.totalEpochs = totalEpochs;
    }
    
    /**
     * Convenience factory method for language model training
     */
    public static ProgressCallback forLanguageModel(boolean detailed) {
        return new ProgressCallback(detailed, true);
    }
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        trainingStartTime = System.currentTimeMillis();
        
        System.out.println("Training started");
        // Training/validation sizes would be provided through metrics
        // For now, we'll just display the start message
        
        if (detailed) {
            System.out.println("═".repeat(80));
        }
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        EpochMetrics epochMetrics = metrics.getEpochMetrics(epoch);
        if (epochMetrics == null) return;
        
        // Simple progress bar
        if (!detailed) {
            printSimpleProgress(epoch, epochMetrics);
        } else {
            printDetailedProgress(epoch, epochMetrics, metrics);
        }
    }
    
    @Override
    public void onTrainingEnd(NeuralNet model, TrainingMetrics metrics) {
        long totalTime = System.currentTimeMillis() - trainingStartTime;
        
        System.out.println("\nTraining completed");
        System.out.printf("Total time: %s%n", formatTime(totalTime));
        
        if (metrics.getBestEpoch() >= 0) {
            System.out.printf("Best epoch: %d (val_acc: %.4f)%n", 
                             metrics.getBestEpoch() + 1, 
                             metrics.getBestValidationAccuracy());
        }
        
        System.out.printf("Final accuracy: %.4f%n", metrics.getFinalAccuracy());
    }
    
    /**
     * Called at the end of each batch (detailed mode only)
     */
    public void onBatchEnd(int batchNum, int totalBatches, BatchTrainer.BatchResult result) {
        if (!detailed) return;
        
        float batchLoss = result.totalLoss / result.batchSize;
        float batchAccuracy = (float) result.correctPredictions / result.batchSize;
        
        // Progress bar for batches
        int barLength = 20;
        int progress = (int) ((float) batchNum / totalBatches * barLength);
        
        System.out.printf("\rBatch %d/%d [%s%s] - loss: %.4f - acc: %.4f",
                         batchNum, totalBatches,
                         "=".repeat(progress),
                         " ".repeat(barLength - progress),
                         batchLoss, batchAccuracy);
        
        if (batchNum == totalBatches) {
            System.out.println(); // New line after last batch
        }
    }
    
    private void printSimpleProgress(int epoch, EpochMetrics metrics) {
        if (languageModelMode) {
            // Show comprehensive stats for language models
            double trainPerplexity = Math.exp(metrics.getTrainingLoss());
            double valPerplexity = Math.exp(metrics.getValidationLoss());
            
            System.out.printf("Epoch %3d - loss: %.4f - acc: %.3f - perplexity: %.1f - val_loss: %.4f - val_acc: %.3f - val_perplexity: %.1f%n",
                             epoch + 1,
                             metrics.getTrainingLoss(),
                             metrics.getTrainingAccuracy(),
                             trainPerplexity,
                             metrics.getValidationLoss(),
                             metrics.getValidationAccuracy(),
                             valPerplexity);
            
            // Add interpretation for very good models
            if (valPerplexity < 150) {
                System.out.println("          [Good model - generating coherent text]");
            } else if (valPerplexity < 500) {
                System.out.println("          [Model improving - keep training]");
            }
        } else {
            // Standard classification metrics
            System.out.printf("Epoch %3d - loss: %.4f - acc: %.4f - val_loss: %.4f - val_acc: %.4f%n",
                             epoch + 1,
                             metrics.getTrainingLoss(),
                             metrics.getTrainingAccuracy(),
                             metrics.getValidationLoss(),
                             metrics.getValidationAccuracy());
        }
    }
    
    private void printDetailedProgress(int epoch, EpochMetrics metrics, TrainingMetrics allMetrics) {
        // Header for new epoch
        System.out.printf("\nEpoch %d/%s%n", epoch + 1, totalEpochs > 0 ? String.valueOf(totalEpochs) : "?");
        
        if (languageModelMode) {
            double trainPerplexity = Math.exp(metrics.getTrainingLoss());
            double valPerplexity = Math.exp(metrics.getValidationLoss());
            
            System.out.printf("├─ Training:   loss: %.4f - acc: %.4f - perplexity: %.1f%n",
                             metrics.getTrainingLoss(),
                             metrics.getTrainingAccuracy(),
                             trainPerplexity);
            
            System.out.printf("├─ Validation: loss: %.4f - acc: %.4f - perplexity: %.1f%n",
                             metrics.getValidationLoss(),
                             metrics.getValidationAccuracy(),
                             valPerplexity);
        } else {
            // Detailed metrics
            System.out.printf("├─ Training:   loss: %.4f - acc: %.4f%n",
                             metrics.getTrainingLoss(),
                             metrics.getTrainingAccuracy());
            
            System.out.printf("├─ Validation: loss: %.4f - acc: %.4f%n",
                         metrics.getValidationLoss(),
                         metrics.getValidationAccuracy());
        }
        
        // Time
        long epochTime = metrics.getEpochTime().toMillis();
        System.out.printf("├─ Time: %s (%d ms/sample)%n",
                         formatTime(epochTime),
                         epochTime / Math.max(1, metrics.getSamplesSeen()));
        
        // Improvement tracking
        if (epoch > 0) {
            EpochMetrics prev = allMetrics.getEpochMetrics(epoch - 1);
            if (prev != null) {
                float lossChange = (float) (metrics.getValidationLoss() - prev.getValidationLoss());
                String trend = lossChange < 0 ? "↓" : lossChange > 0 ? "↑" : "→";
                System.out.printf("└─ Val loss change: %.4f %s%n", Math.abs(lossChange), trend);
            }
        }
    }
    
    private String formatTime(long millis) {
        if (millis < 1000) {
            return millis + "ms";
        } else if (millis < 60000) {
            return String.format("%.1fs", millis / 1000.0);
        } else {
            long minutes = millis / 60000;
            long seconds = (millis % 60000) / 1000;
            return String.format("%dm %ds", minutes, seconds);
        }
    }
}