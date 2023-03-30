package imageshow;

import java.awt.GraphicsDevice;
import java.awt.GraphicsEnvironment;

import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.Pane;
import javafx.scene.layout.VBox;

/**
 * Comic Page with april tags at each corner of the screen
 */
public class ComicPage extends BorderPane {

    // ratio of the page height compared to the monitor height (0.88 for a 27'
    // monitor)
    private final static double PAGE_SCREEN_RATIO = 0.88;
    private final static String APRIL_TAG_PATH = "/tag36_11_%05d.png";
    private final static String PANE_STYLE = "-fx-background-color: black;";
    private final static int APRIL_TAG_SIZE;
    private final static int SCREEN_HEIGHT;
    static {
        GraphicsDevice gd = GraphicsEnvironment.getLocalGraphicsEnvironment()
                .getDefaultScreenDevice();
        SCREEN_HEIGHT = gd.getDisplayMode().getHeight();
        APRIL_TAG_SIZE = 180 * SCREEN_HEIGHT / 1440;
    }

    private final ImageView[] aprilTags;

    public ComicPage(int id, Image comicPage) {
        super();
        if (id > 130)
            throw new IllegalArgumentException(
                    "Id is too big, not enough apriltags");

        setStyle(PANE_STYLE);

        aprilTags = new ImageView[4];
        for (int i = 0; i < 4; i++) {
            // picking 4 apriltags corresponding to each id
            Image tag = new Image(String.format(APRIL_TAG_PATH, 4 * id + i),
                    APRIL_TAG_SIZE, APRIL_TAG_SIZE, true, false);
            aprilTags[i] = new ImageView();
            aprilTags[i].setImage(tag);
            aprilTags[i].setPreserveRatio(true);
            aprilTags[i].setFitWidth(APRIL_TAG_SIZE);
        }

        // set the pane display
        Pane emptyLeft = new Pane();
        Pane emptyRight = new Pane();

        emptyLeft.setPrefHeight(SCREEN_HEIGHT - 2 * APRIL_TAG_SIZE);
        emptyRight.setPrefHeight(SCREEN_HEIGHT - 2 * APRIL_TAG_SIZE);
        Pane left = new VBox(aprilTags[0], emptyLeft, aprilTags[2]);
        this.setLeft(left);
        Pane right = new VBox(aprilTags[1], emptyRight, aprilTags[3]);
        this.setRight(right);
        ImageView comicImage = new ImageView();
        comicImage.setImage(comicPage);
        comicImage.setPreserveRatio(true);
        comicImage.setFitHeight(SCREEN_HEIGHT * PAGE_SCREEN_RATIO);

        this.setCenter(comicImage);
    }

}
