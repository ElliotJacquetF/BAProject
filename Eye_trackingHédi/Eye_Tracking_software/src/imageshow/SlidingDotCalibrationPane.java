package imageshow;

import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;
import java.util.ArrayDeque;
import java.util.Queue;

import javafx.animation.TranslateTransition;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.Pane;
import javafx.util.Duration;

/**
 * Creates a calibration with a marker moving as a pattern
 */
public class SlidingDotCalibrationPane extends Pane {
    private final static String PANE_STYLE = "-fx-background-color: white;";

    private final static String CALIBARATION_MARKER_PATH = "v0.4_calib_marker_02.4b9f83a6.jpg";

    private final int calibrationMarkerSize;
    private boolean calibrationGoing = false;

    private final ImageView marker;

    private Queue<TranslateTransition> transitions;

    private final int height;
    private final int width;

    /**
     * Creates a calibration with a marker moving as a pattern
     */
    public SlidingDotCalibrationPane() {
        setStyle(PANE_STYLE);

        GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment()
                .getDefaultScreenDevice();
        height = gd.getDisplayMode().getHeight();
        width = gd.getDisplayMode().getWidth();
        calibrationMarkerSize = gd.getDisplayMode().getHeight() / 6;
        Image i = new Image(CALIBARATION_MARKER_PATH);
        marker = new ImageView(i);
        marker.setPreserveRatio(true);
        marker.setFitWidth(calibrationMarkerSize);

        getChildren().add(marker);
        marker.setX(width * 30 / 100 - calibrationMarkerSize / 2);
        marker.setY(height * 5 / 100 - calibrationMarkerSize / 2);

        transitions = new ArrayDeque<>();

        // draw the calibration pattern
        addTransition(1, 0, 3);
        // set a small delay before moving
        transitions.peek().setDelay(Duration.seconds(1));
        addTransition(1, 0.2, 1);
        addTransition(0, 0.2, 3);
        addTransition(0, 0.4, 1);
        addTransition(1, 0.4, 3);
        addTransition(1, 0.6, 1);
        addTransition(0, 0.6, 3);
        addTransition(0, 0.8, 1);
        addTransition(1, 0.8, 3);
        addTransition(1, 1, 1);
        addTransition(0, 1, 3);
        addTransition(0, 0, 3);
        addTransition(0.25, 0, 1);
        addTransition(0.25, 1, 3);
        addTransition(0.5, 1, 1);
        addTransition(0.5, 0, 3);
        addTransition(0.75, 0, 1);
        addTransition(0.75, 1, 3);
        addTransition(1, 1, 1);
        addTransition(1, 0, 3);

    }

    private void addTransition(double x, double y, double duration) {

        int relativeMaxX = width * 40 / 100;
        int relativeMaxY = height * 90 / 100;

        TranslateTransition t = new TranslateTransition();
        t.setToX(x * relativeMaxX);
        t.setToY(y * relativeMaxY);

        t.setDuration(Duration.seconds(duration));
        t.setNode(marker);

        // starts the next transition right after the last finished
        t.setOnFinished((e) -> {
            if (calibrationGoing) {
                if (!transitions.isEmpty()) {
                    transitions.poll().play();
                } else {
                    stopCalibration();
                }
            }
        });
        transitions.add(t);
    }

    /**
     * launches the dot animation
     */
    public void startCalibration() {
        calibrationGoing = true;
        marker.setVisible(true);

        transitions.poll().play();

        System.out.println("calibration ...");

    }

    /**
     * stops the dot animation
     */
    public void stopCalibration() {
        System.out.println("stop");
        marker.setVisible(false);
        calibrationGoing = false;

    }
}
