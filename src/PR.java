import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;


public class PR {

    private static final ArrayList<BufferedImage> bufferedImages = new ArrayList<>();

    private static BigDecimal[][][] trainingPixelsImages;
    private static BigDecimal[][] trainingTrueFalse;

//    private static final ArrayList<ArrayList<Integer>> testingPixelsImages = new ArrayList<>();
//    private static final ArrayList<Integer> testingTrueFalse = new ArrayList<>();

    private static BufferedImage dimg_resize;
    private static BufferedImage image_listFilesForFolder;

    public static void main(String[] args) {

        File folder = new File("0");
        listFilesForFolder(folder, bufferedImages);
        int boundary = bufferedImages.size();
        folder = new File("1");
        listFilesForFolder(folder, bufferedImages);
        int bufferedImagesSize = bufferedImages.size();
        trainingPixelsImages = new BigDecimal[bufferedImagesSize][][];
        trainingTrueFalse = new BigDecimal[bufferedImagesSize][];
        for (int i = 0; i < bufferedImagesSize; i++) {

            trainingPixelsImages[i] = new BigDecimal[][] {
                    new BigDecimal[bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth() + 1],
                    new BigDecimal[bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth() + 1],
                    new BigDecimal[bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth() + 1]
            };
            for (int h = 0; h < bufferedImages.get(i).getHeight(); h++) {
                for (int w = 0; w < bufferedImages.get(i).getWidth(); w++) {
                    trainingPixelsImages[i][0][h*bufferedImages.get(i).getHeight() + w] = BigDecimal.valueOf(new Color(bufferedImages.get(i).getRGB(w, h)).getRed()/255.);
                    trainingPixelsImages[i][1][h*bufferedImages.get(i).getHeight() + w] = BigDecimal.valueOf(new Color(bufferedImages.get(i).getRGB(w, h)).getGreen()/255.);
                    trainingPixelsImages[i][2][h*bufferedImages.get(i).getHeight() + w] = BigDecimal.valueOf(new Color(bufferedImages.get(i).getRGB(w, h)).getBlue()/255.);
                }
            }
            // смещение
            trainingPixelsImages[i][0][bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth()] = BigDecimal.ONE;
            trainingPixelsImages[i][1][bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth()] = BigDecimal.ONE;
            trainingPixelsImages[i][2][bufferedImages.get(i).getHeight()*bufferedImages.get(i).getWidth()] = BigDecimal.ONE;

            if (i < boundary)
                trainingTrueFalse[i] = new BigDecimal[] {BigDecimal.ZERO, BigDecimal.ONE};
            else
                trainingTrueFalse[i] = new BigDecimal[] {BigDecimal.ONE, BigDecimal.ZERO};

//            bufferedImages.get(i).flush();
//            bufferedImages.remove(bufferedImages.size() - 1);
        }
        bufferedImages.clear();

        // очистка памяти
        dimg_resize.flush();
        image_listFilesForFolder.flush();
//        pixels.clear();
        if (!folder.delete())
            System.out.println("Folder is not deleted");

        NeuralNetwork appleNN = new NeuralNetwork(trainingPixelsImages, trainingTrueFalse);
    }

    public static void listFilesForFolder(final File folder, ArrayList<BufferedImage> bufferedImages) {
        for (final File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            if (fileEntry.isDirectory())
                listFilesForFolder(fileEntry, bufferedImages);
            else {
                try {
                    image_listFilesForFolder = ImageIO.read(fileEntry);
                    if ( !(image_listFilesForFolder.getWidth() == 100 && image_listFilesForFolder.getHeight() == 100) )
                        image_listFilesForFolder = resize(image_listFilesForFolder, 100, 100);

                    bufferedImages.add(image_listFilesForFolder);

                    image_listFilesForFolder.flush();
                }
                catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        dimg_resize = new BufferedImage(newW, newH, BufferedImage.TYPE_INT_ARGB);

        Graphics2D g2d = dimg_resize.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();

        return dimg_resize;
    }
}
