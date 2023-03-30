package imageshow;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import javafx.geometry.Pos;
import javafx.scene.control.Button;
import javafx.scene.layout.GridPane;
import javafx.scene.text.Text;

/**
 * A Pane showing the question for a given comic page the 3 possible answers.
 *
 */
public class QuestionsPane extends GridPane {

    private final Text question;
    private final String participantPath;

    private static final String TXT_EXTENTION = ".txt";
    private static final String PATH_ANSWERS = "answers/";
    private final static String PANE_STYLE = "-fx-font: 32 Optima;-fx-background-color: white;";

    public QuestionsPane(File f, String participantID) throws IOException {

        participantPath = PATH_ANSWERS + participantID + TXT_EXTENTION;
        setStyle(PANE_STYLE);
        question = new Text();

        List<Button> buttons = new ArrayList<Button>();
        for (int i = 0; i < 3; i++) {
            buttons.add(new Button());
        }

        BufferedReader br = new BufferedReader(new FileReader(f));

        question.setText(br.readLine());
        for (Button b : buttons) {
            b.setText(br.readLine());
            b.setOnMouseClicked(bt -> {
                appendAnswer(f, question.getText(), b.getText());
            });
        }

        br.close();

        Collections.shuffle(buttons);

        GridPane.setRowIndex(question, 0);
        GridPane.setRowIndex(buttons.get(0), 1);
        GridPane.setRowIndex(buttons.get(1), 2);
        GridPane.setRowIndex(buttons.get(2), 3);

        getChildren().addAll(question, buttons.get(0), buttons.get(1),
                buttons.get(2));
        alignmentProperty().set(Pos.CENTER);

    }

    private void appendAnswer(File source, String question, String answer) {
        Path file = Paths.get(participantPath);
        List<String> lines;
        try {
            lines = Arrays.asList(source.getName(), question, answer, "");
            Files.write(file, lines, StandardCharsets.UTF_8,
                    StandardOpenOption.APPEND);
        } catch (NoSuchFileException nf) {
            try {
                lines = Arrays.asList(source.getName(), question, answer, "");
                Files.write(file, lines, StandardCharsets.UTF_8);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Main.nextScene();
    }
}
