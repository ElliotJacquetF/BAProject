package imageshow;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javafx.application.Application;
import javafx.scene.Cursor;
import javafx.scene.Scene;
import javafx.scene.image.Image;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;

public class Main extends Application {

    // set the participant ID before starting the experiment
    public final static String PARTICIPANT_ID = "00000000";

    // texts
    private static final String TEXT_IMPORT_FAILED = "File import failed";

    private static final String TEXT_END = "Fin de l'expérience";

    private static final String TEXT_PAUSE = "Prenez une pause";

    // file paths
    private static final String PATH_QUESTIONS = "/Eye_tracking/questions/";

    private static final String PATH_COMIC_IMAGES = "/Eye_tracking/comic_images/";

    private static final String PATH_TEST_QUESTIONS = "/Eye_tracking/testquestions/";

    private static final String PATH_TESTIMAGES = "/Eye_tracking/testimages/";

    private static final String CALIBRATION_TEXT = "Calibration : ouvrez grand les yeux \net fixez le centre des cercles";

    // slideshow
    private static List<Pane> paneList;

    // current slide
    private static int nb;

    private static Scene mainScene;

    public static void main(String[] args) {
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        paneList = new ArrayList<>();
        nb = 0;

        // generate pane list
        genPaneList();

        mainScene = new Scene(paneList.get(0));

        // go to next Pane when pressing space bar
        mainScene.setOnKeyPressed((ke) -> {
            KeyCode kc = ke.getCode();
            if (kc.equals(KeyCode.SPACE)) {
                if (nb < paneList.size() - 1
                        && !(paneList.get(nb) instanceof QuestionsPane)) {
                    nextScene();
                }
            }
        });

        primaryStage.setScene(mainScene);
        primaryStage.setFullScreen(true);
        primaryStage.show();
    }

    /**
     * Adds every slides to the list
     */
    private static void genPaneList() {

        // CALIBRATION
        paneList.add(new TransitionPane());

        paneList.add(new TextPane(CALIBRATION_TEXT));

        paneList.add(new SlidingDotCalibrationPane());

        // chose one of the test images as first image, doesn't count
        String[] tfList = new File(PATH_TESTIMAGES).list();

        String[] tqList = new File(PATH_TEST_QUESTIONS).list();

        Random r = new Random();
        int k = r.nextInt(tfList.length);

        paneList.add(new TransitionPane());
        paneList.add(new ComicPage(100, new Image(tfList[k])));
        try {
            paneList.add(new QuestionsPane(
                    new File(PATH_TEST_QUESTIONS + tqList[k]), PARTICIPANT_ID));
        } catch (IOException e) {
            System.err.println(TEXT_IMPORT_FAILED);
            e.printStackTrace();
        }

        String[] fList = new File(PATH_COMIC_IMAGES).list();

        String[] qList = new File(PATH_QUESTIONS).list();

        // add all pages in a random order with their respective question
        List<Integer> pageids = new ArrayList<Integer>();
        for (int i = 0; i < fList.length; i++) {
            pageids.add(i);
        }
        Collections.shuffle(pageids);
        addImages(0, pageids.size() / 4, pageids, fList, qList);
        paneList.add(new TransitionPane());
        paneList.add(new TextPane(CALIBRATION_TEXT));
        paneList.add(new SlidingDotCalibrationPane());
        addImages(pageids.size() / 4, pageids.size() / 2, pageids, fList,
                qList);
        paneList.add(new TextPane(TEXT_PAUSE));
        paneList.add(new TransitionPane());
        paneList.add(new TextPane(CALIBRATION_TEXT));
        paneList.add(new SlidingDotCalibrationPane());
        addImages(pageids.size() / 2, 3 * pageids.size() / 4, pageids, fList,
                qList);
        paneList.add(new TransitionPane());
        paneList.add(new TextPane(CALIBRATION_TEXT));
        paneList.add(new SlidingDotCalibrationPane());
        addImages(3 * pageids.size() / 4, pageids.size(), pageids, fList,
                qList);
        paneList.add(new TransitionPane());
        paneList.add(new TextPane(TEXT_END));
    }

    private static void addImages(int startIndex, int endIndex,
            List<Integer> pageids, String[] fList, String[] qList) {
        for (int i = startIndex; i < endIndex; i++) {
            int index = pageids.get(i);
            paneList.add(new TransitionPane());
            Pane p = new ComicPage(index, new Image(fList[index]));
            paneList.add(p);
            try {
                paneList.add(new QuestionsPane(
                        new File(PATH_QUESTIONS + qList[index]),
                        PARTICIPANT_ID));
            } catch (IOException e) {
                System.out.println(TEXT_IMPORT_FAILED);
                e.printStackTrace();
            }
        }
    }

    public static void nextScene() {

        Pane p = paneList.get(++nb);

        // start calibration if calibration
        if (p instanceof SlidingDotCalibrationPane) {
            ((SlidingDotCalibrationPane) p).startCalibration();
        } else if (paneList.get(nb - 1) instanceof SlidingDotCalibrationPane) {
            ((SlidingDotCalibrationPane) paneList.get(nb - 1))
                    .stopCalibration();
        }
        // only show the cursor on questions pane
        if (!(p instanceof QuestionsPane))
            mainScene.setCursor(Cursor.NONE);
        else
            mainScene.setCursor(Cursor.DEFAULT);
        mainScene.setRoot(p);
    };

}
