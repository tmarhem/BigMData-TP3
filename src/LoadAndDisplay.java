import static org.bytedeco.javacpp.opencv_highgui.WINDOW_AUTOSIZE;
import static org.bytedeco.javacpp.opencv_imgproc.compareHist;
import static org.bytedeco.javacpp.opencv_imgproc.*;


import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.namedWindow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.bytedeco.javacpp.opencv_core.CvHistogram;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.JavaCV;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;
import org.opencv.core.Core;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgproc.Imgproc;

public class LoadAndDisplay {

	public static void Show(Mat mat, String title) {
		//
		ToMat converter = new OpenCVFrameConverter.ToMat();
		CanvasFrame canvas = new CanvasFrame(title, 1);
		canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		canvas.showImage(converter.convert(mat));

	}
	
	

	/*
	 * Return histogram for a Mat objet as a array of Float
	 */
	public static Float[] getMyHistogram(Mat image) {
		// HISTOGRAMME
		UByteIndexer idx = (UByteIndexer) image.createIndexer();
		TreeMap<Integer,Integer> tempResults = new TreeMap<Integer,Integer>();
		
		for(int i=0; i<=255; i++) {
			tempResults.put(i, 0);
		}
		
		Float[] results = new Float[256];

		for (int i = 0; i < image.rows(); i++) {
			for (int j = 0; j < image.cols(); j++) {
				int pix = idx.get(i,j);
				if(!tempResults.containsKey(pix)) {
					tempResults.put(pix, 1 );
				} else {
					tempResults.replace(pix, (tempResults.get(pix)+1 ));
				}
			}
		}
		
		System.out.println("HISTOGRAMME A LA MANO");
		for(Entry<Integer,Integer> e : tempResults.entrySet()) {
			//System.out.println(e.getKey()+" "+e.getValue());
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
		
		
		double scale = 0.9 / max(hist)*height;		
		
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

	private static double max(Float[] hist) {
		Float max=(float) 0;
		for(Float f : hist) {
			if (f>max) max=f;
		}
		return max;
	}



	public static void main(String[] args) throws InterruptedException {

		//
		Mat image = imread("data/group.jpg", 1);
		if (image == null || image.empty()) {
			System.out.println("fail");
			return;
		}

		// taille image
		System.out.println("image" + image.cols() + "	x	" + image.rows());
		Show(image, "img");
		//waitKey(0); // Wait for a keystroke in the window

		// FLIP IMAGE
		// Mat flippedImage = imread("data/tower.jpg", 1);
		// flip(image, flippedImage, -1);

		// CIRCLE IMAGE
		/*
		 * Mat imageCircle = imread("data/tower.jpg", 1); circle(imageCircle, // new
		 * Point(420, 150), // 65, // radius new Scalar(0, 200, 0, 0), // 2, // 8, //
		 * 8-connected line 0); // shift
		 * 
		 * opencv_imgproc.putText(imageCircle, // "Lake	and	Tower", // new Point(460,
		 * 200), // FONT_HERSHEY_PLAIN, // 2.0, // new Scalar(0, 255, 0, 3), // 1, // 8,
		 * // false); // Show(imageCircle, "mark");
		 */

		// TEST GETMYHISTOGRAM
		Float[] toPrint = getMyHistogram(image);

		showHistogram(toPrint, "ThisIsHistogram");
		
		//Need MAt conversion
		//Mat backToMat = new Mat();
		//UByteIndexer idx = (UByteIndexer) backToMat.createIndexer();
		
		//Comvertir le format en CV_32F ???
		
		double d = compareHist(image,image,1);
		System.out.println(String.valueOf(d));
		
	    //Mat hist_1 = new Mat();
	   // MatOfFloat ranges = new MatOfFloat(0f, 256f);
	    //MatOfInt histSize = new MatOfInt(25);

	    //Imgproc.calcHist(Arrays.asList(image), new MatOfInt[0], new Mat(), hist_1, histSize, ranges);
		//Float[] toPrint2 = calcHist(image);


	}

}