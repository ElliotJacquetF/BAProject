package imageshow;

import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;

import javafx.concurrent.Task;
import javafx.geometry.Pos;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.GridPane;

/**
 * Calibration Pane that shows 20 consecutive dots on the screen
 *
 * 
 */
public class TwentyDotsCalibrationPane extends GridPane {

    private final static String PANE_STYLE = "-fx-background-color: white;";

    private final static String CALIBARATION_MARKER_PATH = "v0.4_calib_marker_02.4b9f83a6.jpg";
    private final static int MARKER_COUNT = 20;
    private final static int MARKER_TIME = 1500; // ms

    ImageView[] ivs = new ImageView[MARKER_COUNT];

    private final int CALIBRATION_MARKER_SIZE;
    private boolean calibrationGoing = false;

    /**
     * creates the twenty dots calibration
     */
    public TwentyDotsCalibrationPane() {
        setStyle(PANE_STYLE);

        // get screen height
        GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment()
                .getDefaultScreenDevice();
        int height = gd.getDisplayMode().getHeight();
        CALIBRATION_MARKER_SIZE = height / 5;
        Image marker = new Image(CALIBARATION_MARKER_PATH);

        for (int i = 0; i < ivs.length; i++) {
            ivs[i] = new ImageView(marker);
            ivs[i].setPreserveRatio(true);
            ivs[i].setFitWidth(CALIBRATION_MARKER_SIZE);
            ivs[i].setVisible(false);
            GridPane.setConstraints(ivs[i], i % 4, i / 2);
        }
        getChildren().addAll(ivs);
        alignmentProperty().set(Pos.CENTER);

    }

    public void startCalibration() {
        calibrationGoing = true;

        Task<Void> sleeper = new Task<Void>() {
            @Override
            protected Void call() throws Exception {

                for (int i = 0; i < ivs.length; i++) {
                    if (!calibrationGoing)
                        break;
                    ivs[i].setVisible(true);
                    try {
                        Thread.sleep(MARKER_TIME);
                    } catch (InterruptedException e) {
                    }
                    ivs[i].setVisible(false);
                }
                return null;
            }
        };

        Thread t = new Thread(sleeper);
        t.start();

    }

    public void stopCalibration() {
        calibrationGoing = false;

    }

}
