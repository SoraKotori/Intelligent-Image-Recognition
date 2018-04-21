#include <algorithm>
#include <array>
#include <filesystem>
#include <list>
#include <iterator>
#include <string>
#include <utility>
#include <locale>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <type_traits>
#include <opencv2\opencv.hpp>
#include <boost\filesystem.hpp>

template<typename Directory>
decltype(auto) GetFileList(Directory&& _DirectoryName)
{
    boost::filesystem::directory_iterator _Directory(std::forward<Directory>(_DirectoryName));
    std::vector<std::string> _FileList;

    for (const boost::filesystem::path &_Entry : _Directory)
    {
        if ("Thumbs.db" != _Entry.filename())
        {
            auto&& _String = _Entry.string();

            _FileList.push_back(std::forward<decltype(_String)>(_String));
        }
    }

    return _FileList;
}

template<typename File>
auto GetImageSize(File&& _ImagePath)
{
    auto&& _Image = cv::imread(std::forward<File>(_ImagePath));
    auto&& _Pair = std::make_pair(_Image.rows, _Image.cols);

    return _Pair;
}

struct FileControl
{
    boost::filesystem::path _PositivesDirectory{ "Positives" };
    boost::filesystem::path _NegativesDirectory{ "Negatives" };
    boost::filesystem::path _PredictionDirectory{ "Prediction" };

    std::vector<std::string> _PositivesSample = GetFileList(_PositivesDirectory);
    std::vector<std::string> _NegativesSample = GetFileList(_NegativesDirectory);
    std::vector<std::string> _PredictionSample = GetFileList(_PredictionDirectory);

    decltype(auto) AddPositives(cv::Mat _Mat)
    {
        static std::size_t _Index = 0;

        auto _FileName = std::to_string(_Index++) + ".bmp";
        auto _Path = _PositivesDirectory / _FileName;
        auto _String = _Path.string();

        _PositivesSample.push_back(_String);
        cv::imwrite(_String, _Mat);
    }

    decltype(auto) AddNegatives(cv::Mat _Mat)
    {
        static std::size_t _Index = 0;

        auto _FileName = std::to_string(_Index++) + ".bmp";
        auto _Path = _NegativesDirectory / _FileName;
        auto _String = _Path.string();

        _NegativesSample.push_back(_String);
        cv::imwrite(_String, _Mat);
    }
};

template<typename CharacterType = char>
class Process
{
public:
    template<typename... PathType>
    Process(PathType&&... _Args) :
        _ApplicationName(std::forward<PathType>(_Args)...)
    {
    }

    template<typename Type>
    decltype(auto) SetApplicationName(Type&& _Name)
    {
        _ApplicationName = std::forward<Type>(_Name);
    }

    template<typename... PathType>
    decltype(auto) AddArguments(PathType&&... _Args)
    {
        auto AddString = [this](auto&& _Value)
        {
            _ArgumentsStream << ' ' << std::forward<decltype(_Value)>(_Value);
        };

        int _Dummy[] = { 0, ((void)AddString(std::forward<decltype(_Args)>(_Args)), 0) ... };
    }

    decltype(auto) GetCommandLine()
    {
        auto _CommandLine = _ApplicationName + _ArgumentsStream.str();
        return _CommandLine;
    }

    decltype(auto) Run()
    {
        if (_ApplicationName.empty())
        {
            return false;
        }

        auto _CommandLine = GetCommandLine();
        auto _Result = system(_CommandLine.c_str());

        return 0 == _Result ? true : false;
    }

private:
    std::basic_string<CharacterType> _ApplicationName;
    std::basic_ostringstream<CharacterType> _ArgumentsStream;
};

class Haar
{
public:
    Haar() = default;

    template<typename PathType>
    Haar(PathType&& _Path) :
        _Directory(std::forward<PathType>(_Path)),
        _PositiveText(_Directory / "Positive.txt"),
        _NegativeText(_Directory / "Negative.txt"),
        _PositiveVector(_Directory / "Positive.vec"),
        _CascadeXML(_Directory / "cascade.xml")
    {
        if (!boost::filesystem::exists(_Directory))
        {
            boost::filesystem::create_directories(_Directory);
        }

        SetArguments("-data", _Directory.string());
        SetArguments("-info", _PositiveText.string());
        SetArguments("-bg", _NegativeText.string());
        SetArguments("-vec", _PositiveVector.string());

        SetArguments("-weightTrimRate", 0.95);
        SetArguments("-maxDepth ", 1);
        SetArguments("-maxWeakCount ", 100);
        SetArguments("-mode", "ALL");
    }

    template<typename InputIterator>
    decltype(auto) SetPositive(InputIterator _First, InputIterator _Last)
    {
        auto _PositiveSize = std::distance(_First, _Last);
        SetArguments("-num", _PositiveSize);
        SetArguments("-numPos", _PositiveSize * 9 / 10);
        std::ofstream _FileStream(_PositiveText.native());

        std::for_each(_First, _Last, [this, &_FileStream](auto&& _ImagePath)
        {
            auto _Relative = boost::filesystem::relative(_ImagePath, _Directory);
            auto&& _Image = GetImageSize(_ImagePath);

            auto&& _Size = 1;
            auto&& _X = 0;
            auto&& _Y = 0;
            auto&& _Width = _Image.first;
            auto&& _Height = _Image.second;

            _FileStream <<
                _Relative.string() << ' ' <<
                _Size << ' ' <<
                _X << ' ' <<
                _Y << ' ' <<
                _Width << ' ' <<
                _Height << '\n';
        });
    }

    template<typename InputIterator>
    void SetNegative(InputIterator _First, InputIterator _Last)
    {
        SetArguments("-numNeg", std::distance(_First, _Last));
        std::ofstream _FileStream(_NegativeText.native());

        using value_type = typename std::iterator_traits<InputIterator>::value_type;
        std::copy(_First, _Last, std::ostream_iterator<value_type>(_FileStream, "\n"));
    }

    template<typename ArgumentsType, typename Type>
    decltype(auto) SetArguments(ArgumentsType&& _Arguments, Type&& _Value)
    {
        std::ostringstream _Stream;
        _Stream << std::forward<Type>(_Value);

        _ArgumentsMap[std::forward<ArgumentsType>(_Arguments)] = _Stream.str();
    }

    static decltype(auto) GetCreatesamplesSet()
    {
        static const std::set<std::string> _CreatesamplesSet =
        {
            "-info",
            "-vec",
            "-num",
            "-w",
            "-h"
        };

        return _CreatesamplesSet;
    }

    static decltype(auto) GetTraincascadeSet()
    {
        static const std::set<std::string> _TraincascadeSet =
        {
            "-data",
            "-vec",
            "-bg",
            "-numPos",
            "-numNeg",
            "-numStages",
            "-precalcValBufSize",
            "-precalcIdxBufSize",
            "-baseFormatSave",
            "-numThreads",
            "-acceptanceRatioBreakValue",
            "-stageType",
            "-featureType",
            "-w",
            "-h",
            "-bt",
            "-minHitRate",
            "-maxFalseAlarmRate",
            "-weightTrimRate",
            "-maxDepth",
            "-maxWeakCount",
            "-mode"
        };

        return _TraincascadeSet;
    }

    decltype(auto) Training()
    {
        using value_type = typename std::iterator_traits<decltype(std::begin(_ArgumentsMap))>::value_type;

        auto GetSelectList = [this](auto&& _Arguments)
        {
            struct Common
            {
                const std::string &_String;
                Common(const value_type &_Pair) : _String(_Pair.first) { }
                Common(decltype(_String) _String) : _String(_String) { }
            };

            std::list<std::reference_wrapper<value_type>> _List;

            std::set_intersection(
                std::begin(_ArgumentsMap), std::end(_ArgumentsMap),
                std::begin(_Arguments), std::end(_Arguments),
                std::back_inserter(_List),
                [](const Common& Common1, const Common& Common2)
            {
                return Common1._String < Common2._String;
            });

            return _List;
        };

        Process<> _Createsamples{ "opencv_createsamples.exe" };
        for (const value_type& _Pair : GetSelectList(GetCreatesamplesSet()))
        {
            _Createsamples.AddArguments(_Pair.first, _Pair.second);
        }
        _Createsamples.Run();

        Process<> _Traincascade{ "opencv_traincascade.exe" };
        for (const value_type& _Pair : GetSelectList(GetTraincascadeSet()))
        {
            _Traincascade.AddArguments(_Pair.first, _Pair.second);
        }
        _Traincascade.Run();
    }

    decltype(auto) Predicting(cv::Mat _Image)
    {
        std::vector<cv::Rect> _vRect;
        if (boost::filesystem::exists(_CascadeXML))
        {
            cv::CascadeClassifier(_CascadeXML.string()).detectMultiScale(_Image, _vRect);
        }

        return _vRect;
    }

private:
    boost::filesystem::path _Directory;
    boost::filesystem::path _PositiveText;
    boost::filesystem::path _NegativeText;
    boost::filesystem::path _PositiveVector;
    boost::filesystem::path _CascadeXML;

    std::map<std::string, std::string> _ArgumentsMap;
};

decltype(auto) BlurImage(cv::Mat _OriginImage)
{
    cv::Mat _BlurImage;
    cv::Size ksize = cv::Size(20, 20);
    cv::blur(_OriginImage, _BlurImage, ksize);

    return _BlurImage;
}

decltype(auto) GetRectBlurImage(cv::Mat _OriginImage, const cv::Rect &_Rect)
{
    cv::Mat _RectMat;
    _OriginImage(_Rect).copyTo(_RectMat);

    cv::Mat _BlurMat = BlurImage(_OriginImage);
    _RectMat.copyTo(_BlurMat(_Rect));

    return std::make_pair(_RectMat, _BlurMat);
}

decltype(auto) GetImageVector(cv::Mat _OriginImage, const std::vector<cv::Rect> &_vBlock)
{
    cv::Mat _BlurImage = BlurImage(_OriginImage);

    std::vector<cv::Mat> _vRect;
    std::vector<cv::Mat> _vBlur;
    _vRect.reserve(_vBlock.size());
    _vBlur.reserve(_vBlock.size());

    std::for_each(std::begin(_vBlock), std::end(_vBlock), [&](auto&& _Rect)
    {
        cv::Mat _RectMat;
        _OriginImage(_Rect).copyTo(_RectMat);

        cv::Mat _BlurMat = _BlurImage.clone();;
        _RectMat.copyTo(_BlurMat(_Rect));

        _vRect.push_back(_RectMat);
        _vBlur.push_back(_BlurMat);
    });

    return std::make_pair(_vRect, _vBlur);
}

class GUIControl
{
public:
    decltype(auto) GetCurrentPrediction()
    {
        return cv::imread(_FileControl._PredictionSample[_FrameIndex]);
    }

    GUIControl(const cv::String& _wName) :
        _windowsMap(cv::Size(960, 640), CV_8UC3),
        _windowsName(_wName),
        _Haar("Haar"),
        _FrameIndex(44),
        _SelectMethod(1),
        _SelectFlog(false)
    {
        if (_FrameIndex < _FileControl._PredictionSample.size())
        {
            auto&& _Mat = GetCurrentPrediction();
            _Mat.copyTo(_windowsMap(_vectorRect[0]));
        }

        for (std::size_t Index = 1; Index < _White.size(); Index++)
        {
            _White[Index].copyTo(_windowsMap(_vectorRect[Index]));
        }
        _Gray[_SelectMethod].copyTo(_windowsMap(_vectorRect[_SelectMethod]));

        ShowImage();
        cv::setMouseCallback(_wName, onMouse, this);
    }

    decltype(auto) HaarTraining()
    {
        auto&& _List = GetFileList("Haar");
        for (auto&& _File : _List)
        {
            boost::filesystem::remove(_File);
        }

        //_Haar.SetArguments("-w", 21);
        //_Haar.SetArguments("-h", 13);
        _Haar.SetArguments("-numStages", 20);
        _Haar.SetPositive(std::begin(_FileControl._PositivesSample), std::end(_FileControl._PositivesSample));
        _Haar.SetNegative(std::begin(_FileControl._NegativesSample), std::end(_FileControl._NegativesSample));
        _Haar.Training();
    }

    decltype(auto) Predicting(cv::Mat _Mat)
    {
        cv::Rect _Rect;
        if (1 == _SelectMethod)
        {
            auto _vRect = _Haar.Predicting(_Mat);
            if (!_vRect.empty())
            {
                _Rect = _vRect[0];
            }
        }
        else if (2 == _SelectMethod)
        {

        }
        else if (3 == _SelectMethod)
        {

        }

        return _Rect;
    }

    decltype(auto) GetBlock(int x, int y)
    {
        auto is_inside = [](auto&& _Block, auto&& _X, auto&& _Y)
        {
            auto TopLeft = _Block.tl();
            auto BottomRight = _Block.br();

            auto _Left = TopLeft.x;
            auto _Right = BottomRight.x;
            auto _Top = TopLeft.y;
            auto _Bottom = BottomRight.y;

            return
                _Left < _X && _X < _Right &&
                _Top < _Y && _Y < _Bottom ?
                true : false;
        };

        return std::find_if(std::begin(_vectorRect), std::end(_vectorRect), [=](auto&& _Block)
        {
            return is_inside(_Block, x, y);
        });
    }

    decltype(auto) ButtonDown(std::size_t _Index)
    {
        if (0 == _Index)
        {
            if (_SelectFlog)
            {
                auto&& _Mat = GetCurrentPrediction();
                _Mat.copyTo(_windowsMap(_vectorRect[0]));

                _SelectRect.x = _X;
                _SelectRect.y = _Y;
            }
        }
        else if (4 == _Index)
        {
            _Gray[4].copyTo(_windowsMap(_vectorRect[4]));
        }
        else if (5 == _Index)
        {
            _Gray[5].copyTo(_windowsMap(_vectorRect[5]));
        }
        else if (6 == _Index)
        {
            _Gray[6].copyTo(_windowsMap(_vectorRect[6]));
        }
        else if (7 == _Index)
        {
            _Gray[7].copyTo(_windowsMap(_vectorRect[7]));
        }
        else if (9 == _Index)
        {
            _Gray[9].copyTo(_windowsMap(_vectorRect[9]));
        }
    }

    decltype(auto) ButtonUp(std::size_t _Index)
    {
        if (0 == _Index)
        {
            if (_SelectFlog)
            {
                _SelectRect.width = _X - _SelectRect.x;
                _SelectRect.height = _Y - _SelectRect.y;

                auto&& _Mat = GetCurrentPrediction();
                auto&& _Pair = GetRectBlurImage(_Mat, _SelectRect);

                _SelectRectMat = _Pair.first;
                _Pair.second.copyTo(_windowsMap(_vectorRect[0]));
            }
        }
        else if (1 == _Index)
        {
            _Gray[1].copyTo(_windowsMap(_vectorRect[1]));
            _White[2].copyTo(_windowsMap(_vectorRect[2]));
            _White[3].copyTo(_windowsMap(_vectorRect[3]));
            _SelectMethod = 1;
        }
        else if (2 == _Index)
        {
            _White[1].copyTo(_windowsMap(_vectorRect[1]));
            _Gray[2].copyTo(_windowsMap(_vectorRect[2]));
            _White[3].copyTo(_windowsMap(_vectorRect[3]));
            _SelectMethod = 2;
        }
        else if (3 == _Index)
        {
            _White[1].copyTo(_windowsMap(_vectorRect[1]));
            _White[2].copyTo(_windowsMap(_vectorRect[2]));
            _Gray[3].copyTo(_windowsMap(_vectorRect[3]));
            _SelectMethod = 3;
        }
        else if (4 == _Index)
        {
            _FileControl.AddPositives(_SelectRectMat);
            _White[4].copyTo(_windowsMap(_vectorRect[4]));
        }
        else if (5 == _Index)
        {
            _FileControl.AddNegatives(_SelectRectMat);
            _White[5].copyTo(_windowsMap(_vectorRect[5]));
        }
        else if (6 == _Index)
        {
            _White[6].copyTo(_windowsMap(_vectorRect[6]));

            if (0 < _FrameIndex)
            {
                _FrameIndex--;
                auto&& _Mat = GetCurrentPrediction();
                auto _Rect = Predicting(_Mat);

                auto&& _Pair = GetRectBlurImage(_Mat, _Rect);
                _SelectRectMat = _Pair.first;
                auto&& _RectBlur = _Pair.second;
                _RectBlur.copyTo(_windowsMap(_vectorRect[0]));
            }
        }
        else if (7 == _Index)
        {
            _White[7].copyTo(_windowsMap(_vectorRect[7]));

            if (_FrameIndex + 1 < _FileControl._PredictionSample.size())
            {
                _FrameIndex++;
                auto&& _Mat = GetCurrentPrediction();
                auto _Rect = Predicting(_Mat);

                auto&& _Pair = GetRectBlurImage(_Mat, _Rect);
                _SelectRectMat = _Pair.first;
                auto&& _RectBlur = _Pair.second;
                _RectBlur.copyTo(_windowsMap(_vectorRect[0]));
            }
        }
        else if (8 == _Index)
        {
            if (_SelectFlog)
            {
                _White[8].copyTo(_windowsMap(_vectorRect[8]));
                _SelectFlog = !_SelectFlog;
            }
            else
            {
                _Gray[8].copyTo(_windowsMap(_vectorRect[8]));
                _SelectFlog = !_SelectFlog;
            }
        }
        else if (9 == _Index)
        {
            _White[9].copyTo(_windowsMap(_vectorRect[9]));
            if (1 == _SelectMethod)
            {
                HaarTraining();
            }
            else if (2 == _SelectMethod)
            {
                HaarTraining();
            }
            else if (3 == _SelectMethod)
            {
                HaarTraining();
            }
        }
    }

    decltype(auto) MouseEvent(int event, int x, int y, int flags)
    {
        if (CV_EVENT_LBUTTONDOWN == event)
        {
            auto _Iterator = GetBlock(x, y);
            auto _Index = std::distance(std::begin(_vectorRect), _Iterator);
            _X = x;
            _Y = y;

            ButtonDown(_Index);
            ShowImage();
        }
        else if (CV_EVENT_LBUTTONUP == event)
        {
            auto _Iterator = GetBlock(x, y);
            auto _Index = std::distance(std::begin(_vectorRect), _Iterator);
            _X = x;
            _Y = y;

            ButtonUp(_Index);
            ShowImage();
        }
    }

    void ShowImage()
    {
        cv::imshow(_windowsName, _windowsMap);
    }

    cv::Mat _windowsMap;
    cv::String _windowsName;

    cv::Rect _rImage = { 0, 0, 640, 480 };
    cv::Rect _rHaar = { 640, 0, 320, 160 };
    cv::Rect _rSVM = { 640, 160, 320, 160 };
    cv::Rect _rANN = { 640, 320, 320, 160 };
    cv::Rect _rYes = { 0, 480, 240, 160 };
    cv::Rect _rNo = { 240, 480, 240, 160 };
    cv::Rect _rPrev = { 480, 480, 160, 80 };
    cv::Rect _rNext = { 480, 560, 160, 80 };
    cv::Rect _rSelect = { 640, 480, 320, 80 };
    cv::Rect _rTraining = { 640, 560, 320, 80 };

    std::vector<cv::Rect> _vectorRect =
    {
        _rImage,
        _rHaar, _rSVM, _rANN,
        _rYes, _rNo,
        _rPrev, _rNext,
        _rSelect, _rTraining
    };

    std::vector<cv::Mat> _Gray =
    {
        cv::Mat(),
        cv::imread("Gray\\Haar.bmp"),
        cv::imread("Gray\\SVM.bmp"),
        cv::imread("Gray\\ANN.bmp"),
        cv::imread("Gray\\Yes.bmp"),
        cv::imread("Gray\\No.bmp"),
        cv::imread("Gray\\Prev.bmp"),
        cv::imread("Gray\\Next.bmp"),
        cv::imread("Gray\\Select.bmp"),
        cv::imread("Gray\\Training.bmp"),
    };

    std::vector<cv::Mat> _White =
    {
        cv::Mat(),
        cv::imread("White\\Haar.bmp"),
        cv::imread("White\\SVM.bmp"),
        cv::imread("White\\ANN.bmp"),
        cv::imread("White\\Yes.bmp"),
        cv::imread("White\\No.bmp"),
        cv::imread("White\\Prev.bmp"),
        cv::imread("White\\Next.bmp"),
        cv::imread("White\\Select.bmp"),
        cv::imread("White\\Training.bmp"),
    };

    FileControl _FileControl;
    Haar _Haar;
    std::size_t _FrameIndex;
    std::size_t _SelectMethod;
    bool _SelectFlog;
    int _X, _Y;
    cv::Rect _SelectRect;
    cv::Mat _SelectRectMat;

private:

    static void onMouse(int event, int x, int y, int flags, void* userdata)
    {
        static_cast<GUIControl*>(userdata)->MouseEvent(event, x, y, flags);
    }
};

decltype(auto) VideoCapture(const cv::String &_FileName = "V_20171215_234658_vHDR_Auto_OC0.mp4")
{
    cv::VideoCapture _VideoCapture(_FileName);

    std::size_t _Size = 0;
    while (true)
    {
        cv::Mat _Frame;
        if (!_VideoCapture.read(_Frame))
        {
            break;
        }
        cv::resize(_Frame, _Frame, cv::Size(640, 480));
        cv::cvtColor(_Frame, _Frame, CV_BGR2GRAY);

        auto _Path = std::string("Prediction\\") + std::to_string(_Size++) + ".bmp";
        cv::imwrite(_Path, _Frame);
        cv::imshow("windows", _Frame);
        cv::waitKey(1);
    }
}

int main()
{
    cv::String windowname = "windows";
    GUIControl _MouseControl(windowname);

    do
    {
        _MouseControl.ShowImage();
    } while (cv::waitKey(10) != 27);
}
