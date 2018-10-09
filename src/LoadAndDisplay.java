import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import org.bytedeco.javacpp.opencv_core.DMatch;
import org.bytedeco.javacpp.opencv_core.DMatchVector;
import org.bytedeco.javacpp.opencv_core.KeyPointVector;

import static org.bytedeco.javacpp.opencv_imgproc.compareHist;
import static org.bytedeco.javacpp.opencv_features2d.drawKeypoints;
import static org.bytedeco.javacpp.opencv_features2d.drawMatches;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.*;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.Map.Entry;
import java.util.TreeMap;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.opencv_calib3d;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Rect;
import org.bytedeco.javacpp.opencv_core.RectVector;
import org.bytedeco.javacpp.opencv_core.Scalar;
import org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import org.bytedeco.javacpp.opencv_features2d.BFMatcher;
import org.bytedeco.javacpp.opencv_features2d.DrawMatchesFlags;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.bytedeco.javacpp.opencv_shape;
import org.bytedeco.javacpp.opencv_xfeatures2d.SIFT;
import org.bytedeco.javacpp.indexer.UByteIndexer;

import static org.bytedeco.javacpp.opencv_imgproc.calcHist;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter.ToMat;

public class LoadAndDisplay {

	static DMatchVector selectBest(DMatchVector matches, int numberToSelect) {
		DMatch[] sorted = toArray(matches);
		Arrays.sort(sorted, (a, b) -> {
			return a.lessThan(b) ? -1 : 1;
		});
		DMatch[] best = Arrays.copyOf(sorted, numberToSelect);
		return new DMatchVector(best);
	}

	static DMatch[] toArray(DMatchVector matches) {
		assert matches.size() <= Integer.MAX_VALUE;
		int n = (int) matches.size();
//	Convert	keyPoints	to	Scala	sequence
		DMatch[] result = new DMatch[n];
		for (int i = 0; i < n; i++) {
			result[i] = new DMatch(matches.get(i));
		}
		return result;
	}

	public static String getClosestHist(String reference, String[] imagesNamesArray) {
		Mat matReference = imread("data/" + reference + ".jpg", 1);
		matReference = myCalcHist(matReference);

		String bestMatch = null;
		double bestScore = 999999999;

		for (String s : imagesNamesArray) {
			Mat applicant = imread("data/" + s + ".jpg", 1);
			applicant = myCalcHist(applicant);
			if (compareHist(matReference, applicant, 1) < bestScore) {
				bestMatch = String.valueOf(s);
				bestScore = compareHist(matReference, applicant, 1);
			}
			System.out.println("Distance " + reference + "-" + s + " : " + compareHist(matReference, applicant, 1));
		}
		return bestMatch;
	}

	/*
	 * Template matching part
	 */
	public void templateMatching() {
		Mat image1 = imread("data/church01.jpg", 1);
		Mat image2 = imread("data/church03.jpg", 1);
//define	a	template
		Mat target = new Mat(image1);
		Show(target, "Template");
		// define search region
		Mat roi = new Mat(image2);
		// perform template matching
		Mat result = new Mat();
		matchTemplate(roi, // search region
				target, // template
				result, // result
				CV_TM_SQDIFF);
		// similarity measure
		// find most similar location
		double[] minVal = new double[1];
		double[] maxVal = new double[1];
		Point minPt = new Point();
		Point maxPt = new Point();
		// minMaxLoc(result, minVal, maxVal, minPt, maxPt, null);
		// System.out.println("minPt = (" + minPt.x() + ", " + maxPt.y() + ")");
		// draw rectangle at most similar location
		// at minPt in this case
		// rectangle(roi, new Rect(minPt.x(), minPt.y(), target.cols(), target.rows()),
		// new Scalar(255, 255, 255, 0));
		Show(roi, "Best	match");
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
		final int[] channels = new int[] { 0, 1, 2 };
		final Mat mask = new Mat();
		final Mat hist = new Mat();
		final int[] histSize = new int[] { 8, 8, 8 };
		final float[] histRange = new float[] { 0f, 255f };
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
		// for(Entry<Integer,Integer> e : tempResults.entrySet()) {
		// System.out.println(e.getKey()+" "+e.getValue());

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

	private static String compareListHist(Mat image, HashMap<String, Mat> list_image) {
		String nom = "";
		double distance = Integer.MAX_VALUE;
		// System.out.println("valeur de base: " + distance);
		for (Entry<String, Mat> entry : list_image.entrySet()) {
			double d = compareHist(myCalcHist(image), myCalcHist(entry.getValue()), 1);
			// System.out.println(d);
			System.out.println("Distance avec " + entry.getKey() + " " + d);
			if (distance > d) {
				// System.out.println("+1");
				distance = d;
				nom = entry.getKey();
			}
		}
		return "Distance la plus proche :" + nom + " distance : " + distance;

	}

	public static void main(String[] args) throws Exception {

		String[] namesArray = { "baboon1", "baboon2", "baboon3", "baboon4", "boldt", "boldt_salt" };
		LinkedList<String> imagesNames = new LinkedList<String>();
		imagesNames.addAll(Arrays.asList(namesArray));
		HashMap<String, Mat> imagesHash = new HashMap<String, Mat>();

		for (String s : imagesNames) {
			System.out.println(s);
			imagesHash.putIfAbsent(s, imread("data/" + s + ".jpg", 1));
			if (imagesHash.get(s) == null || imagesHash.get(s).empty()) {
				System.out.println("failed to load image " + s);
			}
		}

		////////////////////////////// TEST HISTOGRAM COMP
		histogramComparisonRunTest(imagesHash);
		///////////////////////////////////////

		//////////////////////////////// KEY POINTS DETECTION
		keyPointsRunTest();
		//////////////////////////////////////////////////////

		/////////////////////////////// FACE DETECTION
		faceDetectionRunTest();
		/////////////////////////////////////////////////

		//////////////////////////////// FACE RECOGNITION
		faceRecognitionRunTest();
		////////////////////////////////////////////////

		//////////////////////////////////// OTHERS////////////////////////
		//////////////////////////////////////////////////////////////////
		/////////////////////////// FLIP IMAGE
		// Mat flippedImage = imread("data/tower.jpg", 1);
		// flip(image, flippedImage, -1);
		////////////////////////////////////////////////

		/////////////////////////// CIRCLE IMAGE
		//
		// Mat imageCircle = imread("data/tower.jpg", 1); circle(imageCircle, // new
		// Point(420, 150), // 65, // radius new Scalar(0, 200, 0, 0), // 2, // 8, //
		// 8-connected line 0); // shift
		//
		// opencv_imgproc.putText(imageCircle, // "Lake and Tower", // new Point(460,
		// 200), // FONT_HERSHEY_PLAIN, // 2.0, // new Scalar(0, 255, 0, 3), // 1, //
		/////////////////////////// 8,false); // Show(imageCircle, "mark");
		//
		////////////////////////////////////////////////

	}

	private static void faceRecognitionRunTest() throws Exception {
		CascadeClassifier face_cascade = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");
		String trainingDir = "resources/Groupe-gates.png";
//		FaceRecognizer faceRecognizer = createLBPHFaceRecognizer();
//		FaceRecognizer	faceRecognizer	=	createFisherFaceRecognizer();
//	 	FaceRecognizer	faceRecognizer	= createEigenFaceRecognizer();

		File root = new File(trainingDir);
		FilenameFilter imgFilter = new FilenameFilter() {
			public boolean accept(File dir, String name) {
				name = name.toLowerCase();
				return name.endsWith(".jpg") || name.endsWith(".pgm") || name.endsWith(".png");
			}
		};
		File[] imageFiles = root.listFiles(imgFilter);
		MatVector images = new MatVector(imageFiles.length);

		Mat labels = new Mat(imageFiles.length, 1);
		IntBuffer labelsBuf = labels.createBuffer();
		int counter = 0;
		for (File im : imageFiles) {

			Mat img = imread(im.getAbsolutePath(), CV_LOAD_IMAGE_GRAYSCALE);
			// System.out.println(Integer.parseInt(im.getName().split("-.")[0]));
			int label = Integer.parseInt(im.getName().split("-.")[0]);
			images.put(counter, img);
			labelsBuf.put(counter, label);
			counter++;
//				System.out.println("path:"	+	counter);
		}
	

		images.close();
		labels.close();
	}

	@SuppressWarnings("unused")
	private static void faceDetectionRunTest() throws Exception {

		Mat groupFaces = imread("resources/Group-Faces.jpg");

		CascadeClassifier face_cascade = new CascadeClassifier("resources/haarcascade_frontalface_default.xml");
		CascadeClassifier eye_cascade = new CascadeClassifier("resources/frontalEyes35x16.xml");
		CascadeClassifier smile_cascade = new CascadeClassifier("resources/haarcascade_smile.xml");

		RectVector faces = new RectVector();
		// RectVector eyes = new RectVector();
		// RectVector smiles = new RectVector();

		face_cascade.detectMultiScale(groupFaces, faces);
		// eye_cascade.detectMultiScale(groupFaces, eyes);
		// smile_cascade.detectMultiScale(groupFaces, smiles);

		for (int i = 0; i < faces.size(); i++) {
			Rect face_i = faces.get(i);
			rectangle(groupFaces, face_i, new Scalar(0, 255, 0, 1));
		}

//		for (int i = 0; i < eyes.size(); i++) {
//			Rect eye_i = eyes.get(i);
//			Mat eye = new Mat(groupFaces, eye_i);
//			rectangle(groupFaces, eye_i, new Scalar(255, 0, 0, 1));
//		}

//		for (int i = 0; i < smiles.size(); i++) {
//			Rect smile_i = smiles.get(i);
//			Mat smile = new Mat(groupFaces, smile_i);
//			rectangle(groupFaces, smile_i, new Scalar(0, 0, 255, 1));
//		}

		Show(groupFaces, "face_détction");

		face_cascade.close();
		eye_cascade.close();
		smile_cascade.close();
	}

	@SuppressWarnings("unused")
	private static void keyPointsRunTest() throws Exception {

		Mat image = imread("data/parliament3.bmp", 1);
		Mat image2 = imread("data/parliament1.bmp", 1);
		Mat image3 = imread("data/parliament2.jpg", 1);

		KeyPointVector keyPoints = new KeyPointVector();
		KeyPointVector keyPoints2 = new KeyPointVector();
		KeyPointVector keyPoints3 = new KeyPointVector();

		// params for sift
		int nFeatures = 150;
		int nOctaveLayers = 3;
		double contrastThreshold = 0.01;
		int edgeThreshold = 100;
		double sigma = 1.6;
		Loader.load(opencv_calib3d.class);
		Loader.load(opencv_shape.class);

		SIFT sift, sift2, sift3;

		sift = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
		sift2 = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
		sift3 = SIFT.create(nFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);

		sift.detect(image, keyPoints);
		sift2.detect(image2, keyPoints2);
		sift3.detect(image3, keyPoints3);

		Mat descriptor = new Mat();
		Mat descriptor2 = new Mat();
		Mat descriptor3 = new Mat();

		sift.compute(image, keyPoints, descriptor);
		sift2.compute(image2, keyPoints2, descriptor2);
		sift3.compute(image3, keyPoints3, descriptor3);

		// state : 3 images computed and stored
		BFMatcher matcher = new BFMatcher(5, false);
		DMatchVector matches = new DMatchVector();
		matcher.match(descriptor, descriptor2, matches);
		System.out.println("Matching 1-2 : " + matches.size());
		// selectBest(matches, 10);

		Mat imageMatches = new Mat();
		byte[] mask = null;

		drawMatches(image, keyPoints, image2, keyPoints2, selectBest(matches, 20), imageMatches,
				new Scalar(0, 0, 255, 0), new Scalar(255, 0, 0, 0), mask, DrawMatchesFlags.DEFAULT);
		Show(imageMatches, "imageMatches");

		matcher.close();

		// Drawing output of descriptor
		Mat featureImage = new Mat();
		drawKeypoints(image, keyPoints, featureImage, new Scalar(255, 255, 255, 0),
				DrawMatchesFlags.DRAW_RICH_KEYPOINTS);
		Show(featureImage, "ftImage");
	}

	@SuppressWarnings("unused")
	private static void histogramComparisonRunTest(HashMap<String, Mat> imagesHash) {

		Mat image = imread("data/parliament3.bmp", 1);

		Float[] toPrint = getMyHistogram(image);
		showHistogram(toPrint, "Group");

		double d1 = compareHist(myCalcHist(imagesHash.get("baboon1")), myCalcHist(imagesHash.get("baboon2")), 1);

		System.out.println("Distance baboon1-2 : " + String.valueOf(d1));
		System.out.println();

		String[] applicants = { "baboon1", "baboon2", "baboon3", "group" };
		System.out.println(String.valueOf(getClosestHist("baboon4", applicants)));

		Mat image_compar = imread("data/baboon4.jpg", 1);
		if (image == null || image.empty()) {
			System.out.println("fail");
			return;
		}
		imagesHash.remove("baboon4"); // sinon score = 0

		String leplusproche = compareListHist(image_compar, imagesHash);

		System.out.println(String.valueOf(leplusproche));
	}

}