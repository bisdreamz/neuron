package dev.neuronic.net.training;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * Export and persist training metrics for analysis and comparison.
 * 
 * <p><b>Supported formats:</b>
 * <ul>
 *   <li><b>JSON:</b> Complete metrics with metadata for programmatic analysis</li>
 *   <li><b>CSV:</b> Epoch-by-epoch data for spreadsheet analysis and plotting</li>
 *   <li><b>TensorBoard:</b> Future compatibility for ML tooling integration</li>
 * </ul>
 * 
 * <p><b>Usage - Export Training Results:</b>
 * <pre>{@code
 * // After training with metrics collection
 * TrainingMetrics metrics = new TrainingMetrics();
 * // ... train model and collect metrics ...
 * 
 * // Export comprehensive JSON report
 * MetricsLogger.exportToJson(metrics, "experiment_results.json");
 * 
 * // Export CSV for plotting learning curves  
 * MetricsLogger.exportToCsv(metrics, "training_curves.csv");
 * 
 * // Export with custom metadata
 * Map<String, Object> metadata = Map.of(
 *     "model_type", "SimpleNetInt",
 *     "learning_rate", 0.001,
 *     "batch_size", 32,
 *     "dataset", "MNIST"
 * );
 * MetricsLogger.exportToJson(metrics, "mnist_experiment.json", metadata);
 * }</pre>
 * 
 * <p><b>Progress Callbacks:</b>
 * <pre>{@code
 * // Custom progress callback during training
 * ProgressCallback callback = (epoch, metrics) -> {
 *     if (epoch % 10 == 0) {
 *         logger.info("Epoch {}: accuracy={:.3f}, loss={:.4f}", 
 *                    epoch, metrics.getTrainingAccuracy(), metrics.getTrainingLoss());
 *     }
 * };
 * 
 * // Train with custom callback
 * classifier.trainBulk(data, labels, metrics, callback);
 * }</pre>
 */
public final class MetricsLogger {
    
    /**
     * Callback interface for training progress updates.
     * Allows users to implement custom progress monitoring, logging, or early stopping.
     */
    @FunctionalInterface
    public interface ProgressCallback {
        /**
         * Called after each epoch completes.
         * 
         * @param epochNumber current epoch number (0-indexed)
         * @param epochMetrics metrics for the completed epoch
         */
        void onEpochComplete(int epochNumber, TrainingMetrics.EpochMetrics epochMetrics);
    }
    
    /**
     * Built-in progress callbacks for common use cases.
     */
    public static class ProgressCallbacks {
        
        /**
         * Print progress every N epochs to console.
         */
        public static ProgressCallback printEvery(int interval) {
            return (epoch, metrics) -> {
                if (epoch % interval == 0 || epoch == 0) {
                    System.out.printf("Epoch %d: train_acc=%.3f, val_acc=%.3f, train_loss=%.4f, val_loss=%.4f, time=%dms%n",
                        metrics.getEpochNumber(),
                        metrics.getTrainingAccuracy(),
                        metrics.getValidationAccuracy(), 
                        metrics.getTrainingLoss(),
                        metrics.getValidationLoss(),
                        metrics.getEpochTime().toMillis());
                }
            };
        }
        
        /**
         * Print progress every epoch with detailed stats.
         */
        public static ProgressCallback printDetailed() {
            return (epoch, metrics) -> {
                System.out.printf("Epoch %3d | Train: acc=%6.3f loss=%7.4f | Val: acc=%6.3f loss=%7.4f | %4dms | %dk samples%n",
                    metrics.getEpochNumber(),
                    metrics.getTrainingAccuracy(),
                    metrics.getTrainingLoss(),
                    metrics.getValidationAccuracy(),
                    metrics.getValidationLoss(),
                    metrics.getEpochTime().toMillis(),
                    metrics.getSamplesSeen() / 1000);
            };
        }
        
        /**
         * Silent callback that does nothing (for programmatic use).
         */
        public static ProgressCallback silent() {
            return (epoch, metrics) -> {};
        }
        
        /**
         * Callback that triggers early stopping when validation accuracy stops improving.
         */
        public static ProgressCallback earlyStoppingCallback(int patience, double minImprovement) {
            return new ProgressCallback() {
                private double bestAccuracy = -1.0;
                private int epochsWithoutImprovement = 0;
                
                @Override
                public void onEpochComplete(int epochNumber, TrainingMetrics.EpochMetrics epochMetrics) {
                    double currentAccuracy = epochMetrics.getValidationAccuracy();
                    
                    if (currentAccuracy > bestAccuracy + minImprovement) {
                        bestAccuracy = currentAccuracy;
                        epochsWithoutImprovement = 0;
                        System.out.printf("Epoch %d: New best validation accuracy: %.4f%n", epochNumber, currentAccuracy);
                    } else {
                        epochsWithoutImprovement++;
                        if (epochsWithoutImprovement >= patience) {
                            System.out.printf("Early stopping triggered: no improvement for %d epochs%n", patience);
                            // Note: Actual early stopping would require integration with training loop
                            // This is just a demonstration of the callback pattern
                        }
                    }
                }
            };
        }
        
        /**
         * Combine multiple callbacks into one.
         */
        public static ProgressCallback combine(ProgressCallback... callbacks) {
            return (epoch, metrics) -> {
                for (ProgressCallback callback : callbacks) {
                    try {
                        callback.onEpochComplete(epoch, metrics);
                    } catch (Exception e) {
                        System.err.printf("Error in progress callback: %s%n", e.getMessage());
                    }
                }
            };
        }
    }
    
    private MetricsLogger() {} // Prevent instantiation
    
    /**
     * Export training metrics to JSON format with comprehensive metadata.
     */
    public static void exportToJson(TrainingMetrics metrics, String filePath) throws IOException {
        exportToJson(metrics, Path.of(filePath), Map.of());
    }
    
    /**
     * Export training metrics to JSON format with custom metadata.
     */
    public static void exportToJson(TrainingMetrics metrics, String filePath, Map<String, Object> metadata) throws IOException {
        exportToJson(metrics, Path.of(filePath), metadata);
    }
    
    /**
     * Export training metrics to JSON format with comprehensive metadata.
     */
    public static void exportToJson(TrainingMetrics metrics, Path filePath, Map<String, Object> metadata) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        
        // Add metadata
        json.append("  \"metadata\": {\n");
        json.append("    \"export_time\": \"").append(Instant.now().toString()).append("\",\n");
        json.append("    \"total_epochs\": ").append(metrics.getEpochCount()).append(",\n");
        json.append("    \"total_samples\": ").append(metrics.getTotalSamplesSeen()).append(",\n");
        json.append("    \"training_duration_seconds\": ").append(metrics.getTotalTrainingTime().toSeconds()).append(",\n");
        json.append("    \"samples_per_second\": ").append(String.format("%.2f", metrics.getSamplesPerSecond())).append(",\n");
        json.append("    \"final_accuracy\": ").append(String.format("%.6f", metrics.getFinalAccuracy())).append(",\n");
        json.append("    \"best_validation_accuracy\": ").append(String.format("%.6f", metrics.getBestValidationAccuracy())).append(",\n");
        json.append("    \"best_epoch\": ").append(metrics.getBestEpoch()).append(",\n");
        json.append("    \"overfitting_detected\": ").append(metrics.isOverfitting());
        
        // Add custom metadata
        for (Map.Entry<String, Object> entry : metadata.entrySet()) {
            json.append(",\n    \"").append(entry.getKey()).append("\": ");
            if (entry.getValue() instanceof String) {
                json.append("\"").append(entry.getValue()).append("\"");
            } else {
                json.append(entry.getValue());
            }
        }
        json.append("\n  },\n");
        
        // Add epoch-by-epoch data
        json.append("  \"epochs\": [\n");
        List<TrainingMetrics.EpochMetrics> epochHistory = metrics.getAllEpochMetrics();
        for (int i = 0; i < epochHistory.size(); i++) {
            TrainingMetrics.EpochMetrics epoch = epochHistory.get(i);
            json.append("    {\n");
            json.append("      \"epoch\": ").append(epoch.getEpochNumber()).append(",\n");
            json.append("      \"training_loss\": ").append(String.format("%.6f", epoch.getTrainingLoss())).append(",\n");
            json.append("      \"training_accuracy\": ").append(String.format("%.6f", epoch.getTrainingAccuracy())).append(",\n");
            json.append("      \"validation_loss\": ").append(String.format("%.6f", epoch.getValidationLoss())).append(",\n");
            json.append("      \"validation_accuracy\": ").append(String.format("%.6f", epoch.getValidationAccuracy())).append(",\n");
            json.append("      \"samples_seen\": ").append(epoch.getSamplesSeen()).append(",\n");
            json.append("      \"epoch_time_ms\": ").append(epoch.getEpochTime().toMillis()).append("\n");
            json.append("    }");
            if (i < epochHistory.size() - 1) {
                json.append(",");
            }
            json.append("\n");
        }
        json.append("  ],\n");
        
        // Add summary arrays for easy plotting
        json.append("  \"summary\": {\n");
        json.append("    \"training_loss_history\": ").append(formatDoubleArray(metrics.getTrainingLossHistory())).append(",\n");
        json.append("    \"training_accuracy_history\": ").append(formatDoubleArray(metrics.getTrainingAccuracyHistory())).append(",\n");
        json.append("    \"validation_loss_history\": ").append(formatDoubleArray(metrics.getValidationLossHistory())).append(",\n");
        json.append("    \"validation_accuracy_history\": ").append(formatDoubleArray(metrics.getValidationAccuracyHistory())).append("\n");
        json.append("  }\n");
        
        json.append("}\n");
        
        Files.writeString(filePath, json.toString());
    }
    
    /**
     * Export training metrics to CSV format for spreadsheet analysis.
     */
    public static void exportToCsv(TrainingMetrics metrics, String filePath) throws IOException {
        exportToCsv(metrics, Path.of(filePath));
    }
    
    /**
     * Export training metrics to CSV format for spreadsheet analysis.
     */
    public static void exportToCsv(TrainingMetrics metrics, Path filePath) throws IOException {
        StringBuilder csv = new StringBuilder();
        
        // CSV header
        csv.append("epoch,training_loss,training_accuracy,validation_loss,validation_accuracy,samples_seen,epoch_time_ms\n");
        
        // Data rows
        List<TrainingMetrics.EpochMetrics> epochHistory = metrics.getAllEpochMetrics();
        for (TrainingMetrics.EpochMetrics epoch : epochHistory) {
            csv.append(epoch.getEpochNumber()).append(",");
            csv.append(String.format("%.6f", epoch.getTrainingLoss())).append(",");
            csv.append(String.format("%.6f", epoch.getTrainingAccuracy())).append(",");
            csv.append(String.format("%.6f", epoch.getValidationLoss())).append(",");
            csv.append(String.format("%.6f", epoch.getValidationAccuracy())).append(",");
            csv.append(epoch.getSamplesSeen()).append(",");
            csv.append(epoch.getEpochTime().toMillis()).append("\n");
        }
        
        Files.writeString(filePath, csv.toString());
    }
    
    /**
     * Load training metrics from JSON file.
     */
    public static Map<String, Object> loadFromJson(String filePath) throws IOException {
        return loadFromJson(Path.of(filePath));
    }
    
    /**
     * Load training metrics from JSON file.
     * Note: Returns raw data as Map - would need proper JSON parser for production use.
     */
    public static Map<String, Object> loadFromJson(Path filePath) throws IOException {
        // This is a simplified implementation - in production you'd use Jackson or similar
        String content = Files.readString(filePath);
        
        // For now, just return metadata that can be easily parsed
        Map<String, Object> result = new java.util.HashMap<>();
        result.put("raw_json", content);
        result.put("file_path", filePath.toString());
        result.put("file_size_bytes", Files.size(filePath));
        
        return result;
    }
    
    /**
     * Generate a formatted training report as a string.
     */
    public static String generateReport(TrainingMetrics metrics, Map<String, Object> metadata) {
        StringBuilder report = new StringBuilder();
        
        report.append("=".repeat(60)).append("\n");
        report.append("TRAINING REPORT\n");
        report.append("=".repeat(60)).append("\n\n");
        
        // Metadata section
        if (!metadata.isEmpty()) {
            report.append("EXPERIMENT CONFIGURATION:\n");
            report.append("-".repeat(30)).append("\n");
            for (Map.Entry<String, Object> entry : metadata.entrySet()) {
                report.append(String.format("  %-20s: %s\n", entry.getKey(), entry.getValue()));
            }
            report.append("\n");
        }
        
        // Training summary
        report.append("TRAINING SUMMARY:\n");
        report.append("-".repeat(30)).append("\n");
        report.append(metrics.getSummary()).append("\n\n");
        
        // Performance analysis
        report.append("PERFORMANCE ANALYSIS:\n");
        report.append("-".repeat(30)).append("\n");
        
        double[] trainAcc = metrics.getTrainingAccuracyHistory();
        double[] valAcc = metrics.getValidationAccuracyHistory();
        
        if (trainAcc.length > 0) {
            report.append(String.format("  Initial training accuracy: %.2f%%\n", trainAcc[0] * 100));
            report.append(String.format("  Final training accuracy:   %.2f%%\n", trainAcc[trainAcc.length - 1] * 100));
            report.append(String.format("  Training improvement:      %.2f%% points\n", 
                (trainAcc[trainAcc.length - 1] - trainAcc[0]) * 100));
        }
        
        if (valAcc.length > 0) {
            report.append(String.format("  Initial validation accuracy: %.2f%%\n", valAcc[0] * 100));
            report.append(String.format("  Final validation accuracy:   %.2f%%\n", valAcc[valAcc.length - 1] * 100));
            report.append(String.format("  Validation improvement:      %.2f%% points\n", 
                (valAcc[valAcc.length - 1] - valAcc[0]) * 100));
        }
        
        report.append("\n");
        
        // Recent epochs summary
        if (metrics.getEpochCount() > 0) {
            report.append("RECENT EPOCHS:\n");
            report.append("-".repeat(30)).append("\n");
            List<TrainingMetrics.EpochMetrics> epochs = metrics.getAllEpochMetrics();
            int start = Math.max(0, epochs.size() - 5); // Last 5 epochs
            
            for (int i = start; i < epochs.size(); i++) {
                TrainingMetrics.EpochMetrics epoch = epochs.get(i);
                report.append(String.format("  %s\n", epoch.toString()));
            }
        }
        
        report.append("\n").append("=".repeat(60)).append("\n");
        
        return report.toString();
    }
    
    /**
     * Print a formatted training report to console.
     */
    public static void printReport(TrainingMetrics metrics, Map<String, Object> metadata) {
        System.out.println(generateReport(metrics, metadata));
    }
    
    /**
     * Print a basic training report to console.
     */
    public static void printReport(TrainingMetrics metrics) {
        printReport(metrics, Map.of());
    }
    
    // Helper methods
    
    private static String formatDoubleArray(double[] array) {
        if (array.length == 0) return "[]";
        
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < array.length; i++) {
            sb.append(String.format("%.6f", array[i]));
            if (i < array.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}