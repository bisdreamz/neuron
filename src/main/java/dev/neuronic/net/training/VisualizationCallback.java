package dev.neuronic.net.training;

import dev.neuronic.net.NeuralNet;
import dev.neuronic.net.training.TrainingMetrics.EpochMetrics;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Visualization callback that generates training curve plots.
 * 
 * Features:
 * - Plots training and validation loss curves
 * - Plots training and validation accuracy curves
 * - Saves plots as PNG images
 * - Updates plots in real-time during training
 * - Supports custom styling and colors
 */
public class VisualizationCallback implements TrainingCallback {
    
    private final String outputDirectory;
    private final boolean saveFinal;
    private final boolean saveEveryEpoch;
    private final int updateFrequency; // Update plot every N epochs
    
    // Plot data
    private final List<Float> trainLosses = new ArrayList<>();
    private final List<Float> valLosses = new ArrayList<>();
    private final List<Float> trainAccuracies = new ArrayList<>();
    private final List<Float> valAccuracies = new ArrayList<>();
    
    // Plot settings
    private static final int PLOT_WIDTH = 800;
    private static final int PLOT_HEIGHT = 600;
    private static final int MARGIN = 80;
    private static final Color TRAIN_COLOR = new Color(30, 144, 255); // Dodger blue
    private static final Color VAL_COLOR = new Color(255, 99, 71); // Tomato
    private static final Color LR_COLOR = new Color(34, 139, 34); // Forest green
    private static final Color GRID_COLOR = new Color(220, 220, 220);
    
    public VisualizationCallback(String outputDirectory) {
        this(outputDirectory, true, false, 5);
    }
    
    public VisualizationCallback(String outputDirectory, boolean saveFinal, 
                                boolean saveEveryEpoch, int updateFrequency) {
        this.outputDirectory = outputDirectory;
        this.saveFinal = saveFinal;
        this.saveEveryEpoch = saveEveryEpoch;
        this.updateFrequency = updateFrequency;
    }
    
    @Override
    public void onTrainingStart(NeuralNet model, TrainingMetrics metrics) {
        // Clear previous data
        trainLosses.clear();
        valLosses.clear();
        trainAccuracies.clear();
        valAccuracies.clear();
        
        // Create output directory
        File dir = new File(outputDirectory);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        System.out.println("Visualization: plots will be saved to " + outputDirectory);
    }
    
    @Override
    public void onEpochEnd(int epoch, TrainingMetrics metrics) {
        EpochMetrics epochMetrics = metrics.getEpochMetrics(epoch);
        if (epochMetrics == null) return;
        
        // Collect data
        trainLosses.add((float) epochMetrics.getTrainingLoss());
        valLosses.add((float) epochMetrics.getValidationLoss());
        trainAccuracies.add((float) epochMetrics.getTrainingAccuracy());
        valAccuracies.add((float) epochMetrics.getValidationAccuracy());
        
        // Update plots if needed
        boolean shouldUpdate = saveEveryEpoch || 
                              ((epoch + 1) % updateFrequency == 0);
        
        if (shouldUpdate) {
            try {
                generatePlots(epoch + 1);
            } catch (IOException e) {
                System.err.println("Failed to generate plots: " + e.getMessage());
            }
        }
    }
    
    @Override
    public void onTrainingEnd(NeuralNet model, TrainingMetrics metrics) {
        if (saveFinal) {
            try {
                generatePlots(trainLosses.size());
                System.out.println("Final training plots saved to " + outputDirectory);
            } catch (IOException e) {
                System.err.println("Failed to generate final plots: " + e.getMessage());
            }
        }
    }
    
    private void generatePlots(int currentEpoch) throws IOException {
        // Generate loss plot
        BufferedImage lossPlot = createPlot(
            "Training History - Loss",
            "Epoch",
            "Loss",
            trainLosses,
            valLosses,
            null,
            true
        );
        
        // Generate accuracy plot
        BufferedImage accuracyPlot = createPlot(
            "Training History - Accuracy",
            "Epoch",
            "Accuracy",
            trainAccuracies,
            valAccuracies,
            null,
            false
        );
        
        // Save plots
        String suffix = saveEveryEpoch ? "_epoch_" + currentEpoch : "";
        ImageIO.write(lossPlot, "PNG", new File(outputDirectory, "loss" + suffix + ".png"));
        ImageIO.write(accuracyPlot, "PNG", new File(outputDirectory, "accuracy" + suffix + ".png"));
    }
    
    private BufferedImage createPlot(String title, String xLabel, String yLabel,
                                    List<Float> trainData, List<Float> valData,
                                    List<Float> thirdData, boolean logarithmic) {
        BufferedImage image = new BufferedImage(PLOT_WIDTH, PLOT_HEIGHT, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        
        // Enable antialiasing
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        
        // White background
        g2d.setColor(Color.WHITE);
        g2d.fillRect(0, 0, PLOT_WIDTH, PLOT_HEIGHT);
        
        // Calculate plot area
        int plotLeft = MARGIN;
        int plotRight = PLOT_WIDTH - MARGIN;
        int plotTop = MARGIN;
        int plotBottom = PLOT_HEIGHT - MARGIN;
        int plotWidth = plotRight - plotLeft;
        int plotHeight = plotBottom - plotTop;
        
        // Find data ranges
        float minY = Float.MAX_VALUE;
        float maxY = Float.MIN_VALUE;
        
        for (Float val : trainData) {
            minY = Math.min(minY, val);
            maxY = Math.max(maxY, val);
        }
        
        if (valData != null) {
            for (Float val : valData) {
                minY = Math.min(minY, val);
                maxY = Math.max(maxY, val);
            }
        }
        
        // Add padding to Y range
        float yPadding = (maxY - minY) * 0.1f;
        minY -= yPadding;
        maxY += yPadding;
        
        // Draw grid
        g2d.setColor(GRID_COLOR);
        g2d.setStroke(new BasicStroke(1));
        
        // Vertical grid lines
        for (int i = 0; i <= 10; i++) {
            int x = plotLeft + (plotWidth * i / 10);
            g2d.drawLine(x, plotTop, x, plotBottom);
        }
        
        // Horizontal grid lines
        for (int i = 0; i <= 10; i++) {
            int y = plotTop + (plotHeight * i / 10);
            g2d.drawLine(plotLeft, y, plotRight, y);
        }
        
        // Draw axes
        g2d.setColor(Color.BLACK);
        g2d.setStroke(new BasicStroke(2));
        g2d.drawLine(plotLeft, plotBottom, plotRight, plotBottom); // X axis
        g2d.drawLine(plotLeft, plotTop, plotLeft, plotBottom); // Y axis
        
        // Draw data
        g2d.setStroke(new BasicStroke(2));
        
        // Training data
        if (!trainData.isEmpty()) {
            g2d.setColor(TRAIN_COLOR);
            drawDataLine(g2d, trainData, plotLeft, plotTop, plotWidth, plotHeight, 
                        trainData.size(), minY, maxY);
        }
        
        // Validation data
        if (valData != null && !valData.isEmpty()) {
            g2d.setColor(VAL_COLOR);
            drawDataLine(g2d, valData, plotLeft, plotTop, plotWidth, plotHeight, 
                        trainData.size(), minY, maxY);
        }
        
        // Third data (e.g., learning rate)
        if (thirdData != null && !thirdData.isEmpty()) {
            g2d.setColor(LR_COLOR);
            drawDataLine(g2d, thirdData, plotLeft, plotTop, plotWidth, plotHeight, 
                        thirdData.size(), minY, maxY);
        }
        
        // Draw title
        g2d.setColor(Color.BLACK);
        g2d.setFont(new Font("Arial", Font.BOLD, 18));
        FontMetrics fm = g2d.getFontMetrics();
        int titleWidth = fm.stringWidth(title);
        g2d.drawString(title, (PLOT_WIDTH - titleWidth) / 2, 30);
        
        // Draw axis labels
        g2d.setFont(new Font("Arial", Font.PLAIN, 14));
        
        // X label
        fm = g2d.getFontMetrics();
        int xLabelWidth = fm.stringWidth(xLabel);
        g2d.drawString(xLabel, (PLOT_WIDTH - xLabelWidth) / 2, PLOT_HEIGHT - 20);
        
        // Y label (rotated)
        g2d.rotate(-Math.PI / 2);
        int yLabelWidth = fm.stringWidth(yLabel);
        g2d.drawString(yLabel, -(PLOT_HEIGHT + yLabelWidth) / 2, 20);
        g2d.rotate(Math.PI / 2);
        
        // Draw legend
        if (valData != null || thirdData != null) {
            drawLegend(g2d, plotRight - 150, plotTop + 20, valData != null, thirdData != null);
        }
        
        g2d.dispose();
        return image;
    }
    
    private void drawDataLine(Graphics2D g2d, List<Float> data, int plotLeft, int plotTop,
                             int plotWidth, int plotHeight, int maxPoints,
                             float minY, float maxY) {
        if (data.size() < 2) return;
        
        int prevX = plotLeft;
        int prevY = plotTop + plotHeight - (int) ((data.get(0) - minY) / (maxY - minY) * plotHeight);
        
        for (int i = 1; i < data.size(); i++) {
            int x = plotLeft + (i * plotWidth / (maxPoints - 1));
            int y = plotTop + plotHeight - (int) ((data.get(i) - minY) / (maxY - minY) * plotHeight);
            
            g2d.drawLine(prevX, prevY, x, y);
            
            prevX = x;
            prevY = y;
        }
    }
    
    private void drawLegend(Graphics2D g2d, int x, int y, boolean hasVal, boolean hasThird) {
        int lineHeight = 20;
        int lineLength = 30;
        
        // Background
        int legendHeight = lineHeight * (1 + (hasVal ? 1 : 0) + (hasThird ? 1 : 0)) + 10;
        g2d.setColor(new Color(255, 255, 255, 200));
        g2d.fillRect(x - 5, y - 5, 140, legendHeight);
        g2d.setColor(Color.BLACK);
        g2d.drawRect(x - 5, y - 5, 140, legendHeight);
        
        // Legend items
        g2d.setFont(new Font("Arial", Font.PLAIN, 12));
        
        // Training
        g2d.setColor(TRAIN_COLOR);
        g2d.setStroke(new BasicStroke(2));
        g2d.drawLine(x, y + lineHeight / 2, x + lineLength, y + lineHeight / 2);
        g2d.setColor(Color.BLACK);
        g2d.drawString("Training", x + lineLength + 5, y + lineHeight / 2 + 5);
        y += lineHeight;
        
        // Validation
        if (hasVal) {
            g2d.setColor(VAL_COLOR);
            g2d.drawLine(x, y + lineHeight / 2, x + lineLength, y + lineHeight / 2);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Validation", x + lineLength + 5, y + lineHeight / 2 + 5);
            y += lineHeight;
        }
        
        // Third line
        if (hasThird) {
            g2d.setColor(LR_COLOR);
            g2d.drawLine(x, y + lineHeight / 2, x + lineLength, y + lineHeight / 2);
            g2d.setColor(Color.BLACK);
            g2d.drawString("Learning Rate", x + lineLength + 5, y + lineHeight / 2 + 5);
        }
    }
}