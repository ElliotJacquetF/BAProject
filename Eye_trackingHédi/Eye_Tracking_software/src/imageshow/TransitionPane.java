package imageshow;

import javafx.scene.layout.Pane;

/**
 * A black Pane
 *
 */
public class TransitionPane extends Pane {
    private static final String TRANSITION_PANE_STYLE = "-fx-background-color: black;";

    /**
     * Creates a black Pane
     */
    public TransitionPane() {
        super();
        setStyle(TRANSITION_PANE_STYLE);
    }
}
