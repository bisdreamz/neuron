package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.training.TrainingMetrics.EpochMetrics;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Model checkpoint callback that saves the model during training.
 * 
 * Features:
 * - Save best model based on validation metrics
 * - Save model at regular intervals
 * - Configurable file naming patterns
 * - Automatic directory creation
 */
public class ModelCheckpointCallback implements TrainingCallback {
    
    private final String filepath;
    private final String monitor;
    private final boolean saveOnlyBest;
    private final int saveFrequency; // Save every N epochs (0 = only on improvement)
    
    // State tracking
    private float bestValue;
    private boolean monitorIncreasing;
    private int lastSaveEpoch = -1;
    
    /**
     * Creates checkpoint callback that saves best model based on validation accuracy.
     */
    public ModelCheckpointCallback(String filepath) {
        this(filepath, "val_accuracy", true, 0);
    }
    
    /**
     * Full constructor with all options.
     * 
     * @param filepath path pattern for saving (can include {epoch} and {val_accuracy} placeholders)
     * @param monitor metric to monitor for best model ("val_loss" or "val_accuracy")
     * @param saveOnlyBest if true, only save when monitored metric improves
     * @param saveFrequency save every N epochs (0 = only on improvement)
     */
    public ModelCheckpointCallback(String filepath, String monitor, 
                                  boolean saveOnlyBest, int saveFrequency) {
        this.filepath = filepath;
        this.monitor = monitor;
        this.saveOnlyBest = saveOnlyBest;
        this.saveFrequency = saveFrequency;
        
        this.monitorIncreasing = monitor.equals("val_accuracy");
    }
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        // Initialize best value
        bestValue = monitorIncreasing ? Float.NEGATIVE_INFINITY : Float.POSITIVE_INFINITY;
        
        // Ensure directory exists
        Path path = Paths.get(filepath);
        Path parent = path.getParent();
        if (parent != null) {
            try {
                Files.createDirectories(parent);
            } catch (IOException e) {
                System.err.println("Warning: Could not create checkpoint directory: " + e.getMessage());
            }
        }
        
        System.out.printf("Model checkpointing: saving to %s (monitoring %s)%n", 
                         filepath, monitor);
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        EpochMetrics epochMetrics = metrics.getEpochMetrics(epoch);
        if (epochMetrics == null) return;
        
        // Get monitored value
        float currentValue = monitor.equals("val_loss") 
            ? (float) epochMetrics.getValidationLoss() 
            : (float) epochMetrics.getValidationAccuracy();
        
        // Check if we should save
        boolean shouldSave = false;
        String reason = "";
        
        if (saveOnlyBest) {
            // Save only if improved
            boolean improved = monitorIncreasing 
                ? currentValue > bestValue 
                : currentValue < bestValue;
            
            if (improved) {
                bestValue = currentValue;
                shouldSave = true;
                reason = String.format("new best %s: %.4f", monitor, currentValue);
            }
        } else {
            // Always save, but track if it's the best
            shouldSave = true;
            boolean isBest = monitorIncreasing 
                ? currentValue > bestValue 
                : currentValue < bestValue;
            
            if (isBest) {
                bestValue = currentValue;
                reason = String.format("new best %s: %.4f", monitor, currentValue);
            } else {
                reason = "periodic save";
            }
        }
        
        // Check frequency constraint
        if (saveFrequency > 0 && (epoch - lastSaveEpoch) < saveFrequency) {
            shouldSave = false;
        }
        
        // Save if needed
        if (shouldSave) {
            String filename = formatFilename(filepath, epoch, epochMetrics);
            
            try {
                // Save the model (we need to get it from somewhere)
                // For now, we'll need to pass it through the callback system
                saveModel(filename, metrics);
                lastSaveEpoch = epoch;
                
                System.out.printf("Model checkpoint saved: %s (%s)%n", filename, reason);
            } catch (IOException e) {
                System.err.printf("Failed to save checkpoint: %s%n", e.getMessage());
            }
        }
    }
    
    private String formatFilename(String pattern, int epoch, EpochMetrics metrics) {
        String result = pattern;
        
        // Replace placeholders
        result = result.replace("{epoch}", String.format("%03d", epoch + 1));
        result = result.replace("{val_loss}", String.format("%.4f", metrics.getValidationLoss()));
        result = result.replace("{val_accuracy}", String.format("%.4f", metrics.getValidationAccuracy()));
        result = result.replace("{loss}", String.format("%.4f", metrics.getTrainingLoss()));
        result = result.replace("{accuracy}", String.format("%.4f", metrics.getTrainingAccuracy()));
        
        return result;
    }
    
    protected void saveModel(String filename, TrainingMetrics metrics) throws IOException {
        // Note: We need access to the model to save it
        // This is a limitation of the current callback design
        // In practice, we might need to enhance the callback interface
        // or use a different approach for model access
        
        // For now, just create an empty file as a placeholder
        Path path = Paths.get(filename);
        Files.createFile(path);
    }
    
    /**
     * Enhanced version that can actually save the model.
     * Requires model reference to be passed during construction.
     */
    public static class WithModel extends ModelCheckpointCallback {
        private final NeuralNet model;
        
        public WithModel(NeuralNet model, String filepath) {
            super(filepath);
            this.model = model;
        }
        
        public WithModel(NeuralNet model, String filepath, String monitor, 
                        boolean saveOnlyBest, int saveFrequency) {
            super(filepath, monitor, saveOnlyBest, saveFrequency);
            this.model = model;
        }
        
        @Override
        protected void saveModel(String filename, TrainingMetrics metrics) throws IOException {
            model.save(Paths.get(filename));
        }
    }
}