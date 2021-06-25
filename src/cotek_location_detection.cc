#include "cotek_location_detection.h"

#include <iostream>
namespace cotek_detect {
// 是红色阈值
constexpr int kIsRedThreshold = 200;
// 确定一条直线至少需要多少曲线相交，400
constexpr int kMinCrossLineTh = 300;
// 闭操作核大小
constexpr int kCloseKernelSize = 15;
// 清理文件夹后剩余图片数量
constexpr int kRemainImageNumber = 50;
// 清理文件夹周期
constexpr int kClearImagesPeriod = 12;
// 前景特征点数量是背景特征点数量的几倍，超过阈值认为有货物
constexpr int kIsOccupiedLevelTh = 3;
// 直线最小长度，小于该长度丢弃，50
constexpr double kMinLineLength = 100;
// 同一方向上两条线段判定为一条线段的最大允许断裂间隔，20
constexpr double kMaxLineGap = 30;
// 最小面积
constexpr double kMinAreaTh = 6000;
// 最小匹配汉明距离
constexpr double kMinMatchedHanmingDistance = 30.;
// 差分方式，不同部分占整体面积百分比，超过该阈值认为有货物
constexpr double kIsOccupiedRatioTh = 0.3;
// 获取图片超时时间 s
constexpr double kCaptureTimeout = 5;
// 图片保存文件夹路径
constexpr char kImageSavedFolder[] =
    "/home/fengsc/location_detection/image_saved/section_";

LocationDetect::LocationDetect()
    : error_type_(ErrorType::NORMAL),
      calibration_finished_(false),
      detection_finished_(false) {
  executor_ =
      std::make_shared<std::thread>(&LocationDetect::ProcessImage, this);
}

LocationDetect::~LocationDetect() {
  if (executor_) {
    executor_->join();
    executor_ = nullptr;
  }
}

void LocationDetect::SetErrorType(ErrorType e) { error_type_ = e; }

void LocationDetect::ClearErrorType() { error_type_ = ErrorType::NORMAL; }

ErrorType LocationDetect::GetErrorState() { return error_type_; }

bool LocationDetect::IsCalibrated() { return calibration_finished_; }

bool LocationDetect::IsDetected() { return detection_finished_; }

// 清除图片
void LocationDetect::ClearImages(const std::string &path) {
  std::vector<cv::String> files_temp;
  cv::glob(path, files_temp);
  std::list<cv::String> files(files_temp.begin(), files_temp.end());
  while (files.size() > kRemainImageNumber) {
    std::remove(files.front().c_str());
    files.pop_front();
  }
  std::cout << "Deleted " << files_temp.size() - files.size()
            << " images, remaining " << files.size() << " images!" << std::endl;
}

// 记录时间
const std::string LocationDetect::GetTimeStampNow() {
  auto stamp =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  tm *tt = localtime(&stamp);
  std::string name_timed =
      std::to_string(tt->tm_year + 1900) + "_" +
      std::to_string(tt->tm_mon + 1) + "_" + std::to_string(tt->tm_mday) + "_" +
      std::to_string(tt->tm_hour) + "_" + std::to_string(tt->tm_min) + "_" +
      std::to_string(tt->tm_sec);
  // std::cout << "name_timed : " << name_timed << std::endl;
  return name_timed;
}

bool LocationDetect::Start(std::string url, OperationMode mode) {
  ClearErrorType();
  int num = 0;
  std::string folder;
  if (section_.count(url) == 0) {
    // 以url id的最后一位位编号
    int pos1 = url.find_last_of('.');
    int pos2 = url.find_last_of('/');
    auto s = url.substr(pos1 + 1, pos2 - pos1 - 1).c_str();
    num = atoi(s);

    // 新建该url对应的文件夹
    folder = kImageSavedFolder + std::to_string(num);
    std::string command = "mkdir -p " + folder;
    system(command.c_str());
  }
  // 打开视频流
  capture_ = std::make_shared<cv::VideoCapture>(url);
  do {
    capture_->open(url);
    std::cout << "Opening " << url << " camera..." << std::endl;
  } while (!capture_->isOpened());
  std::cout << "Open camera succeed!" << std::endl;

  cv::Mat frame;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  while (!capture_->read(frame)) {
    std::cout << "Capturing background image..." << std::endl;
    if (std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - t0)
            .count() > kCaptureTimeout) {
      std::cout << "Error : Capture image timeout!" << std::endl;
      SetErrorType(ErrorType::LOAD_IMAGE_ERROR);
      return false;
    } else {
      ClearErrorType();
    }
  }

  if (mode == OperationMode::CALIBRATION) {
    if (section_.count(url) == 0) {
      Section sect;
      sect.background = frame.clone();
      sect.id = num;
      sect.folder_path = folder;
      section_.emplace(url, sect);
    }
    start_calibration_ = true;
  } else if (mode == OperationMode::DETECTION) {
    if (section_.count(url) == 0) {
      std::cout << "Can't start detection of the URL, please start calibration "
                   "firstly!"
                << std::endl;
      SetErrorType(ErrorType::NO_BACKGROUND);
      start_detection_ = false;
      return false;
    }
    ClearErrorType();
    section_.at(url).foreground = frame.clone();
    start_detection_ = true;
  } else {
    std::cout << "Operation mode illegal, mode : " << static_cast<uint>(mode)
              << std::endl;
  }

  current_url_ = url;
  return true;
}

// 获取标定后图片路径
bool LocationDetect::GetCalibratedImagePath(std::string url,
                                            std::string &output) {
  if (section_.count(url)) {
    output = section_.at(url).calibrated_image_path;
    calibration_finished_ = false;
    return true;
  }
  if (IsCalibrated()) {
    SetErrorType(ErrorType::URL_UNCALIBRATED);
    return false;
  }
  return false;
}

// 获取库区内库位状态
bool LocationDetect::GetSectionCeilState(std::string url,
                                         std::vector<CeilState> &state) {
  if (section_.count(url)) {
    ClearErrorType();
    state = section_.at(url).ceil_state;
    return true;
  }
  SetErrorType(ErrorType::URL_UNCALIBRATED);
  return false;
}

// 对图像滤波
void LocationDetect::ImageFilter(cv::Mat &src, bool use_median_blur,
                                 int median_block_size, bool use_box_filter,
                                 int box_size, bool use_guass_blur,
                                 int guass_size, int sigmaX) {
  if (use_median_blur) {
    cv::medianBlur(src, src, median_block_size);
  }
  if (use_box_filter) {
    cv::boxFilter(src, src, -1, cv::Size(box_size, box_size));
  }
  if (use_guass_blur) {
    cv::GaussianBlur(src, src, cv::Size(guass_size, guass_size), sigmaX, 0);
  }
}

// 剪切ROI，区域外填充黑色，存入map
void LocationDetect::SectionSegmentation(cv::Mat input,
                                         std::map<int, cv::Mat> &idx_stn) {
  try {
    auto current_section = section_.at(current_url_);
    idx_stn.clear();
    for (int i = 0; i < current_section.ceil_number; i++) {
      // 黑色image，存放单个库位
      cv::Mat image = cv::Mat::zeros(input.size(), CV_8UC3);
      cv::Mat mask = image.clone();
      // 单个库位ROI
      drawContours(mask, current_section.ceil_contours, i, cv::Scalar::all(255),
                   CV_FILLED);
      input.copyTo(image, mask);
      idx_stn.emplace(i, image(current_section.ceil_rect.at(i)));
      // std::string ceil_name = "section " +
      //                         std::to_string(current_section.id) + " ceil
      //                         " + std::to_string(i);
      // cv::imshow(ceil_name, idx_stn.at(i));
    }
  } catch (cv::Exception e) {
    std::cout << e.what() << std::endl;
  }
}

//标定过程
void LocationDetect::ProcessCalibration(cv::Mat src) {
  // 直方图均衡
  imshow("origin image", src);
  normalize(src, src, 0, 255, cv::NORM_MINMAX);
  imshow("normalized image", src);
  ImageFilter(src, true, 7, false, 0, true, 5, 5);
  imshow("filterd image", src);

  // 分离BGR通道
  std::vector<cv::Mat> BGR_channel;
  cv::split(src, BGR_channel);
  cv::Mat channel_R = BGR_channel[2].clone();
  imshow("channel red", channel_R);
  cv::threshold(channel_R, channel_R, kIsRedThreshold, 255, cv::THRESH_BINARY);
  imshow("threshold red", channel_R);

  // 去噪，提取线段
  std::vector<cv::Vec4f> lines_Hough;  // 线段端点
  HoughLinesP(channel_R, lines_Hough, 1, CV_PI / 180, kMinCrossLineTh,
              kMinLineLength, kMaxLineGap);
  cv::Mat img_hough = cv::Mat::zeros(src.size(), CV_8UC1);
  for (size_t i = 0, n = lines_Hough.size(); i < n; i++) {
    cv::Vec4i point_i = lines_Hough[i];
    line(img_hough, cv::Point(point_i[0], point_i[1]),
         cv::Point(point_i[2], point_i[3]), cv::Scalar(255), 2, cv::LINE_AA);
  }
  // imshow("image Hough", img_hough);

  // 计算库位边缘点，画边界
  auto current_section = section_.at(current_url_);
  current_section.ceil_contours.clear();
  current_section.ceil_rect.clear();
  current_section.ceil_center.clear();
  std::vector<std::vector<cv::Point>> contours_all;
  std::vector<cv::Vec4i> hierarchy;

  findContours(img_hough, contours_all, hierarchy, cv::RETR_LIST,
               cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  if (contours_all.size() < 1) {
    std::cout << "Can't find contours, please check input image!" << std::endl;
    return;
  }

  // 剔除小区域
  for (int i = 0, n = contours_all.size(); i < n; i++) {
    if (contourArea(contours_all[i]) < kMinAreaTh) continue;
    current_section.ceil_contours.push_back(contours_all[i]);
  }
  if (current_section.ceil_contours.size() < 1) {
    std::cout << "Error : ceil contours number : 0" << std::endl;
    return;
  }
  current_section.ceil_number = current_section.ceil_contours.size() - 1;
  for (int i = 0; i < current_section.ceil_number; i++) {
    cv::RotatedRect rotate_rect =
        cv::minAreaRect(current_section.ceil_contours[i]);
    current_section.ceil_rect.emplace_back(
        boundingRect(current_section.ceil_contours.at(i)));
    current_section.ceil_center.emplace_back(rotate_rect.center);
  }

  // 分割库位
  SectionSegmentation(current_section.background.clone(),
                      current_section.background_ceil);

  cv::Mat dst = current_section.background.clone();
  for (int i = 0; i < current_section.ceil_number; i++) {
    drawContours(dst, current_section.ceil_contours, i, cv::Scalar::all(255), 3,
                 cv::LINE_AA, cv::noArray(), 0, cv::Point());
    //  库位编号
    putText(dst, std::to_string(i + 1), current_section.ceil_center.at(i),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar::all(255), 2, cv::LINE_AA,
            false);
  }

  current_section.calibrated_image_path =
      current_section.folder_path + "/section_" +
      std::to_string(current_section.id) + "_" + GetTimeStampNow() + ".png";
  if (!cv::imwrite(current_section.calibrated_image_path, dst)) {
    std::cout << "Error: write image failed! Please check path" << std::endl;
    SetErrorType(ErrorType::WRITE_IMAGE_ERROR);
    calibration_finished_ = false;
    return;
  }

  ClearErrorType();
  calibration_finished_ = true;
  start_calibration_ = false;
  std::cout << "Section Calibration completed! Image saved to path: "
            << current_section.calibrated_image_path << std::endl;
  std::string display_name =
      "calibrated secton " + std::to_string(current_section.id);
  cv::imshow(display_name, dst);
}

// 模型加载及初始化过程
void LocationDetect::CreatModule(){
  torch::DeviceType device_type = at::KCPU;
  torch::jit::script::Module module = torch::jit::load("../cpu.pth", device_type);
}

// 图像预处理过程
void LocationDetect::Mat2tensor(cv::Mat &src, torch::Tensor &output){
  cv::resize(src, src, {224, 224});
  cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
  src.convertTo(src, CV_32F, 1.0 / 255.0);
  torch::TensorOptions option(torch::kFloat32);
  auto img_tensor = torch::from_blob(src.data, {1, img.rows, img.cols, img.channels()}, option);
  img_tensor = img_tensor.permute({0, 3, 1, 2});
  
  // 均值归一化
  img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
	img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
	img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

  output = img_tensor.clone();
}

// 分类检测库位状态
const std::vector<CeilState> LocationDetect::MobilenetV3Detection(){
  std::vector<CeilState> state;
  // 得到想要的库区所有的库位数量
  int n = section_.at(current_url_).ceil_number;
  if(n < 1){
    std::cout << "Class extract error: no corrent ceil_number" << std::endl;
  }

  for(int i = 0; i < n; i++){
    // 获取当前库位的实时图像
    cv::Mat ceil_img = section_.at(current_url_).foreground_ceil.at(i).clone();
  }
}

// ORB提取检测库位状态
const std::vector<CeilState> LocationDetect::ORBExtractDetection() {
  std::vector<CeilState> state;
  int n = section_.at(current_url_).ceil_number;
  if (n < 1) {
    std::cout << "ORB extract error: no background!" << std::endl;
    return state;
  }

  for (int i = 0; i < n; i++) {
    cv::Mat ceil_bg = section_.at(current_url_).background_ceil.at(i).clone();
    cv::Mat ceil_fg = section_.at(current_url_).foreground_ceil.at(i).clone();

    if (ceil_bg.data == nullptr || ceil_fg.data == nullptr) {
      std::cout << "background or foreground date empty!" << std::endl;
      state.emplace_back(CeilState::UNKNOWN);
      continue;
    }

    // imshow("background", ceil_bg);
    // imshow("foreground", ceil_fg);

    std::vector<cv::KeyPoint> keypoints_bg, keypoints_fg;
    cv::Mat descriptors_bg, descriptors_fg;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher =
        cv::DescriptorMatcher::create("BruteForce-Hamming");
    // 检测 Oriented FAST 角点位置
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(ceil_bg, keypoints_bg);
    detector->detect(ceil_fg, keypoints_fg);
    int keypoints_bg_nums = keypoints_bg.size();
    int keypoints_fg_nums = keypoints_fg.size();

    std::cout << "background keypoint numbers: " << keypoints_bg_nums
              << ", foreground keypoint numbers: " << keypoints_fg_nums
              << std::endl;
    if (keypoints_bg_nums == 0 || keypoints_bg_nums == 0) {
      std::cout << "background or foreground key points numbers : 0, state "
                   "unknown"
                << std::endl;
      state.emplace_back(CeilState::UNKNOWN);
      continue;
    }
    if (keypoints_fg_nums / keypoints_bg_nums > kIsOccupiedLevelTh) {
      state.emplace_back(CeilState::BUSY);
      continue;
    }
    // 计算 BRIEF 描述子
    descriptor->compute(ceil_bg, keypoints_bg, descriptors_bg);
    descriptor->compute(ceil_fg, keypoints_fg, descriptors_fg);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    // 描述子匹配，使用 Hamming 距离
    std::vector<cv::DMatch> matches;
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    // 在descriptors_fg中搜索匹配descriptors_bg的特征点
    matcher->match(descriptors_bg, descriptors_fg, matches);
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    // 匹配点对筛选
    auto min_max =
        minmax_element(matches.begin(), matches.end(),
                       [](const cv::DMatch &m1, const cv::DMatch &m2) {
                         return m1.distance < m2.distance;
                       });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_bg.rows; i++) {
      if (matches[i].distance <=
          std::max(2 * min_dist, kMinMatchedHanmingDistance)) {
        good_matches.push_back(matches[i]);
      }
    }
    int good_matches_nums = good_matches.size();

    std::cout << "Available matched numbers between background and foreground: "
              << good_matches_nums << std::endl;

    if (good_matches_nums * 1.0 / keypoints_fg_nums < 0.1) {
      state.emplace_back(CeilState::BUSY);
    } else {
      state.emplace_back(CeilState::FREE);
    }

    std::cout << "orb detected result state[" << i
              << "]: " << static_cast<int>(state.at(i)) << std::endl;

    // cv::waitKey(20);

#if 0  // for debug
    double extract_time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();
    std::cout << "extract ORB cost = " << extract_time_used << " seconds. "
              << std::endl;

    cv::Mat outimg_bg;
    drawKeypoints(ceil_bg, keypoints_bg, outimg_bg, cv::Scalar::all(-1),
                  cv::DrawMatchesFlags::DEFAULT);
    imshow("bg ORB features", outimg_bg);

    double match_time_used =
        std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3)
            .count();
    std::cout << "match ORB cost = " << match_time_used << " seconds. "
              << std::endl;

    int matched_bg2fg_nums = matches.size();
    std::cout
        << " the number of features in background matched with foreground: "
        << matched_bg2fg_nums << std::endl;
    std::cout << "Max distance: " << max_dist << std::endl;
    std::cout << "Min distance: " << min_dist << std::endl;

    cv::Mat img_match;
    cv::Mat img_goodmatch;
    drawMatches(ceil_bg, keypoints_bg, ceil_fg, keypoints_fg, matches, img_match);
    drawMatches(ceil_bg, keypoints_bg, ceil_fg, keypoints_fg, good_matches,
                img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
#endif
  }

  return state;
}

// 差分检测库位状态
const std::vector<CeilState> LocationDetect::DifferenceDetection() {
  std::vector<CeilState> state;
  int n = section_.at(current_url_).ceil_number;
  if (n < 1) {
    std::cout << "ORB extract error: no background!" << std::endl;
    return state;
  }

  for (int i = 0; i < n; i++) {
    cv::Mat ceil_bg = section_.at(current_url_).background_ceil.at(i).clone();
    cv::Mat ceil_fg = section_.at(current_url_).foreground_ceil.at(i).clone();
    // cv::imshow("ceil background", ceil_bg);
    // cv::imshow("ceil foreground", ceil_fg);

    if (ceil_bg.data == nullptr || ceil_fg.data == nullptr) {
      std::cout << "background or foreground date empty!" << std::endl;
      state.emplace_back(CeilState::UNKNOWN);
      continue;
    }

    // 去除亮斑
    cv::illuminationChange(ceil_bg, ceil_bg.clone(), ceil_bg, 0.1, 2);
    cv::illuminationChange(ceil_fg, ceil_fg.clone(), ceil_fg, 0.1, 2);
    // cv::imshow("background low light", ceil_bg);
    // cv::imshow("foreground low light", ceil_fg);

    // 去噪，模糊
    ImageFilter(ceil_bg, true, 5, true, 9, true, 9, 10);
    ImageFilter(ceil_fg, 1, 5, 1, 9, 1, 9, 10);
    // cv::imshow("background filtered", ceil_bg);
    // cv::imshow("foreground filtered", ceil_fg);

    // BGR转HSV
    cv::Mat HSV_bg, HSV_fg;
    cv::cvtColor(ceil_bg, HSV_bg, cv::COLOR_BGR2HSV);
    cv::cvtColor(ceil_fg, HSV_fg, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> HSV_bg_channel, HSV_fg_channel;
    cv::split(HSV_bg, HSV_bg_channel);
    cv::split(HSV_fg, HSV_fg_channel);
    // cv::imshow("background H", HSV_bg_channel[0]);
    // cv::imshow("background S", HSV_bg_channel[1]);
    // cv::imshow("foreground H", HSV_fg_channel[0]);
    // cv::imshow("foreground S", HSV_fg_channel[1]);

    // 前、后景差分
    cv::Mat H_diff, S_diff, V_diff;
    cv::absdiff(HSV_fg_channel[0], HSV_bg_channel[0], H_diff);
    cv::absdiff(HSV_fg_channel[1], HSV_bg_channel[1], S_diff);
    cv::absdiff(HSV_fg_channel[2], HSV_bg_channel[2], V_diff);
    // cv::imshow("H diff", H_diff);
    // cv::imshow("S diff", S_diff);
    // cv::imshow("V diff", V_diff);

    // 合并H,S通道
    cv::Mat HS;
    cv::add(H_diff, S_diff, HS);
    // cv::imshow("HS merged", HS);

    // 过滤微亮杂点
    cv::threshold(HS, HS, 25, 255, cv::THRESH_TOZERO);
    // cv::imshow("threshold TOZERO", HS);
    cv::threshold(HS, HS, 5, 255, cv::THRESH_OTSU);
    // cv::imshow("threshold OTSU", HS);
    cv::Mat HS_close,
        dilate_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(kCloseKernelSize, kCloseKernelSize));
    cv::morphologyEx(HS, HS_close, cv::MORPH_CLOSE, dilate_kernel);
    // cv::imshow("merge zero tosu close", HS_close);

    // 均值
    cv::Scalar HS_mean = cv::mean(HS);
    std::cout << "Occupied pixel ratio: " << HS_mean[0] / 255. << std::endl;

    if (HS_mean[0] / 255. > kIsOccupiedRatioTh) {
      state.emplace_back(CeilState::BUSY);
    } else {
      state.emplace_back(CeilState::FREE);
    }

    std::cout << "diff detected result state[" << i
              << "]: " << static_cast<int>(state.at(i)) << std::endl;
  }
  return state;
}

// 整体检测过程
void LocationDetect::ProcessDetection(cv::Mat src) {
  SectionSegmentation(src, section_.at(current_url_).foreground_ceil);

  const std::vector<CeilState> ORB_detect_result = ORBExtractDetection();

  const std::vector<CeilState> diff_detect_result = DifferenceDetection();

  // std::cout << "orb result size : " << ORB_detect_result.size()
  //           << ", diff result size : " << diff_detect_result.size()
  //           << std::endl;

  if (ORB_detect_result.size() != diff_detect_result.size()) {
    std::cout << "ORB result and differ result not equal!" << std::endl;
    SetErrorType(ErrorType::DETECTION_ERROR);
  } else {
    ClearErrorType();
  }

  auto current_section = section_.at(current_url_);
  if (GetErrorState() == ErrorType::NORMAL) {
    current_section.ceil_state.clear();
    for (int i = 0, n = current_section.ceil_number; i < n; i++) {
      // std::cout << ORB_detect_result[i] << ", " << ORB_detect_result[i]
      //           << std::endl;
      if (ORB_detect_result[i] == CeilState::UNKNOWN ||
          diff_detect_result[i] == CeilState::UNKNOWN) {
        current_section.ceil_state.emplace_back(CeilState::UNKNOWN);
      } else if (ORB_detect_result[i] == CeilState::BUSY &&
                 diff_detect_result[i] == CeilState::BUSY) {
        current_section.ceil_state.emplace_back(CeilState::BUSY);
      } else {
        current_section.ceil_state.emplace_back(CeilState::FREE);
      }
    }
    detection_finished_ = true;
    start_calibration_ = false;

#if 1  // for debug
    std::cout << "section state : " << std::endl;
    for (int i = 0, n = current_section.ceil_number; i < n; i++) {
      std::cout << static_cast<int>(current_section.ceil_state[i]) << " ";
    }
    std::cout << std::endl;
#endif
  }
}

// 图像预处理
void LocationDetect::ProcessImage() {
  try {
    while (1) {
      // 标定库位
      if (start_calibration_) {
        std::cout << "******* Starting section calibration ******* "
                  << std::endl;
        ProcessCalibration(section_.at(current_url_).background.clone());
      }

      std::cout << std::endl;

      //   库位检测
      if (start_detection_) {
        std::cout << "******* Starting section detection ******* " << std::endl;
        ProcessDetection(section_.at(current_url_).foreground.clone());
      }

      cv::waitKey(20);

      // 定期清理历史图片
      auto start_t = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::hours>(
              std::chrono::steady_clock::now() - start_t)
              .count() > kClearImagesPeriod) {
        for (auto it = section_.begin(), itend = section_.end(); it != itend;
             it++) {
          ClearImages((*it).second.folder_path);
        }
      }

      // std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  } catch (cv::Exception e) {
    std::cout << e.what() << std::endl;
  }
}

}  // namespace cotek_detect