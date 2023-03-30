package imageshow;

import javafx.scene.layout.BorderPane;
import javafx.scene.text.Text;

/**
 * A grey Pane with the given text in the middle
 *
 */
public class TextPane extends BorderPane {
    private Text breakText;
    private final static String PANE_STYLE = "-fx-font: 32 Optima;-fx-background-color: darkgrey;";

    /**
     * Creates a new Pane with the given text in the middle
     * 
     * @param txt
     */
    public TextPane(String txt) {
        this.setStyle(PANE_STYLE);
        breakText = new Text();
        breakText.setText(txt);
        setCenter(breakText);
    }
}
