package exploration;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class RobotExploration {
    public static void main(String[] args) throws Exception {
        ///////////////////////////// ////// CONFIGURATION ////// /////////////////////////////
        // Zone de la prise de vue
        Rectangle screenRect = new GetWindowRect().getWindowRect();
        // Temps en secondes de la vidéo
        int duree = 5;
        // Nombres d'images par seconde
        int frequence = 10;
        // Début des noms de fichiers (ils seront complétés par _{indice}.png)
        String tmpDirectory = System.getProperty("java.io.tmpdir") + "Robot\\robot";
        File prefix = new File(tmpDirectory);
        System.out.println("write into" + tmpDirectory);
        Robot robot = new Robot();
        int max = duree * frequence;
        // nombre d'images au total
        long delai = 1000 / frequence;
        // temps d'attentes entre deux images (en ms)
        // tableau d'enregistrement des images
        BufferedImage[] images = new BufferedImage[max];
        System.out.println("Début des captures d'images.");
        for (int i = 0; i < max; i++) {
            // capture de l'image
            images[i] = robot.createScreenCapture(screenRect);
            // attente entre deux images à capturer
            Thread.sleep(delai);
        }
        System.out.println("Fin des captures d'images.");
        // nom générique des images
        String format = String.format("%s_%%0%dd.png", prefix.getAbsolutePath(), String.valueOf(max).length());
        System.out.println("Début d'enregistrement des images --> " + format);
        for (int i = 0; i < max; i++) {
            // nom spécifique de l'image
            String file = String.format(format, i);
            // enregistrement de l'image
            ImageIO.write(images[i], "png", new File(file));
        }
        System.out.println("Fin d'enregistrement des images.");
    }

}