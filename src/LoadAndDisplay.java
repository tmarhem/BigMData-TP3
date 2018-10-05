import static org.bytedeco.javacpp.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.namedWindow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.bytedeco.javacpp.opencv_core.CvHistogram;
import org.bytedeco.javacv.JavaCV;

import static org.bytedeco.javacpp.opencv_imgproc.compareHist;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Map.Entry;
import java.util.TreeMap;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;

public class LoadAndDisplay {

	public static String getClosestHist(String reference, String[] imagesNamesArray) {
		Mat matReference = imread("data/" + reference + ".jpg", 1);	
		matReference = myCalcHist(matReference);
		
		String bestMatch = null;
		double bestScore = 999999999;
		
		for(String s : imagesNamesArray) {
			Mat applicant = imread("data/" + s + ".jpg", 1);	
			applicant = myCalcHist(applicant);
			if(compareHist(matReference, applicant,1)<bestScore) {
				bestMatch = String.valueOf(s);
				bestScore = compareHist(matReference, applicant,1);
			}
			System.out.println("Distance "+reference+"-"+s+" : "+compareHist(matReference, applicant,1));
		}
		return bestMatch;
	}
	
	/*
	 * Displays image
	 */
	public static void Show(Mat mat, String title) {
		//
		ToMat converter = new OpenCVFrameConverter.ToMat();
		CanvasFrame canvas = new CanvasFrame(title, 1);
		canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		canvas.showImage(converter.convert(mat));

	}

	
	/*
	 * Return Mat histogram from openCV type, in 3D
	 */
	public static Mat myCalcHist(Mat image) {

        // Compute histogram
        final int[] channels = new int[]{0, 1, 2};
        final Mat mask = new Mat();
        final Mat hist = new Mat();
        final int[] histSize = new int[]{8, 8, 8};
        final float[] histRange = new float[]{0f, 255f};
        IntPointer intPtrChannels = new IntPointer(channels);
        IntPointer intPtrHistSize = new IntPointer(histSize);
        final PointerPointer<FloatPointer> ptrPtrHistRange = new PointerPointer<>(histRange, histRange, histRange);
        calcHist(image, 1, intPtrChannels, mask, hist, 3, intPtrHistSize, ptrPtrHistRange, true, false);
        return hist;
    }


	/*
	 * Return histogram for a Mat objet as a array of Float
	 */
	public static Float[] getMyHistogram(Mat image) {
		// HISTOGRAMME
		UByteIndexer idx = (UByteIndexer) image.createIndexer();
		TreeMap<Integer, Integer> tempResults = new TreeMap<Integer, Integer>();

		for (int i = 0; i <= 255; i++) {
			tempResults.put(i, 0);
		}

		Float[] results = new Float[256];

		for (int i = 0; i < image.rows(); i++) {
			for (int j = 0; j < image.cols(); j++) {
				int pix = idx.get(i, j);
				if (!tempResults.containsKey(pix)) {
					tempResults.put(pix, 1);
				} else {
					tempResults.replace(pix, (tempResults.get(pix) + 1));
				}
			}
		}

		System.out.println("HISTOGRAMME A LA MANO");
		//for(Entry<Integer,Integer> e : tempResults.entrySet()) {
			//System.out.println(e.getKey()+" "+e.getValue());

		for (Entry<Integer, Integer> e : tempResults.entrySet()) {
			System.out.println(e.getKey() + " " + e.getValue());
			results[e.getKey()] = e.getValue().floatValue();
		}
		return results;
	}

	/*
	 * Draw Histogram
	 */
	public static void showHistogram(Float[] hist, String caption) {
		int numberOfBins = 256;
		// Output image size
		int width = numberOfBins;
		int height = numberOfBins;
		// Set highest point to 90% of the number of bins

		double scale = 0.9 / max(hist) * height;

		// Create a color image to draw on
		BufferedImage canvas = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = canvas.createGraphics();
		// Paint background
		g.setPaint(Color.WHITE);
		g.fillRect(0, 0, width, height);
		// Draw a vertical line for each bin
		g.setPaint(Color.BLUE);
		for (int bin = 0; bin < numberOfBins; bin++) {
			int h = (int) Math.round(hist[bin] * scale);
			g.drawLine(bin, height - 1, bin, height - h - 1);
		}
		// Cleanup
		g.dispose();
		// Create an image and show the histogram
		CanvasFrame canvasF = new CanvasFrame(caption, 1);
		canvasF.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		canvasF.showImage(canvas);
	}

	/*
	 * Returns highest peak in an histogram
	 */
	private static double max(Float[] hist) {
		Float max = (float) 0;
		for (Float f : hist) {
			if (f > max)
				max = f;
		}
		return max;
	}
	
	private static String compareListHist(Mat image, HashMap<String,Mat> list_image) {
		String nom = "";
		double distance = Integer.MAX_VALUE;
		//System.out.println("valeur de base: " + distance);
		for (Entry<String, Mat> entry : list_image.entrySet()) {		
			double d = compareHist(myCalcHist(image),myCalcHist(entry.getValue()),1);
			//System.out.println(d);
			System.out.println("Distance avec " + entry.getKey() + " "+ d);
			if(distance > d) {
				//System.out.println("+1");
				distance = d;
				nom = entry.getKey();
			}
		}
		return "Distance la plus proche :" + nom + " distance : " + distance;
		
	}

	
	public static void main(String[] args) throws InterruptedException {

		String[] namesArray = {"baboon1", "baboon2", "baboon3", "baboon4","boldt","boldt_salt"};
		LinkedList<String> imagesNames = new LinkedList<String>();
		imagesNames.addAll(Arrays.asList(namesArray));
		HashMap<String,Mat> imagesHash = new HashMap<String,Mat>();
		
		for(String s : imagesNames) {
			System.out.println(s);
			imagesHash.putIfAbsent(s, imread("data/" + s + ".jpg", 1));		
			if(imagesHash.get(s)==null || imagesHash.get(s).empty()) 
			{
				System.out.println("failed to load image "+s);
			}
		}
		
		Mat image = imread("data/group.jpg", 1);
		if (image == null || image.empty()) 
		{
			System.out.println("fail");
			return;
		}


		// taille image
//		System.out.println("image" + image.cols() + "	x	" + image.rows());
//		Show(image, "img");
//		Show(imagesHash.get("boldt"), "boldt");
//		Show(imagesHash.get("baboon1"), "baboon");
//

		// TEST GETMYHISTOGRAM
		
		Float[] toPrint = getMyHistogram(image);
		showHistogram(toPrint, "Group");
		
		double d1 = compareHist(myCalcHist(imagesHash.get("baboon1")),myCalcHist(imagesHash.get("baboon2")),1);

		System.out.println("Distance baboon1-2 : " + String.valueOf(d1));
		System.out.println();

		
		String[] applicants = { "baboon1", "baboon2", "baboon3", "group"};
		System.out.println(String.valueOf( getClosestHist( "baboon4" , applicants )));


		Mat image_compar = imread("data/baboon4.jpg", 1);
		if (image == null || image.empty()) 
		{
			System.out.println("fail");
			return;
		}
		imagesHash.remove("baboon4"); //sinon score = 0
		
		String leplusproche = compareListHist(image_compar, imagesHash);
		
		System.out.println(String.valueOf(leplusproche));
		
		/////////////////////////// FLIP IMAGE
		// 		Mat flippedImage = imread("data/tower.jpg", 1);
		// 		flip(image, flippedImage, -1);
		////////////////////////////////////////////////
		
		/////////////////////////// CIRCLE IMAGE
		//		
		//		 Mat imageCircle = imread("data/tower.jpg", 1); circle(imageCircle, // new
		//		 Point(420, 150), // 65, // radius new Scalar(0, 200, 0, 0), // 2, // 8, //
		//		 8-connected line 0); // shift
		//		  
		//		 opencv_imgproc.putText(imageCircle, // "Lake	and	Tower", // new Point(460,
		//		 200), // FONT_HERSHEY_PLAIN, // 2.0, // new Scalar(0, 255, 0, 3), // 1, // 8,false); // Show(imageCircle, "mark");
		//		 
		////////////////////////////////////////////////
		
	}

	
}