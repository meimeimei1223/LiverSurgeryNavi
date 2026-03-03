// SimpleCamera.hpp
// License: MIT (自由に使用可能)
// Header-only webcam capture library for Linux (V4L2) and Windows (DirectShow)
#pragma once

#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>

//=============================================================================
// プラットフォーム判定
//=============================================================================
#if defined(_WIN32)
#define SIMPLE_CAMERA_WINDOWS
#elif defined(__linux__)
#define SIMPLE_CAMERA_LINUX
#elif defined(__APPLE__)
#define SIMPLE_CAMERA_MACOS
#endif

//=============================================================================
// Linux実装 (V4L2)
//=============================================================================
#ifdef SIMPLE_CAMERA_LINUX

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <linux/videodev2.h>
#include <dirent.h>
#include <errno.h>

class SimpleCamera {
private:
    int m_fd = -1;
    int m_width = 0;
    int m_height = 0;
    bool m_isOpened = false;
    uint32_t m_pixelFormat = 0;

    struct Buffer {
        void* start = nullptr;
        size_t length = 0;
    };
    std::vector<Buffer> m_buffers;
    std::vector<unsigned char> m_rgbBuffer;

    static int xioctl(int fd, unsigned long request, void* arg) {
        int r;
        do { r = ioctl(fd, request, arg); } while (r == -1 && errno == EINTR);
        return r;
    }

    // YUYV → RGB変換
    void convertYUYVtoRGB(const unsigned char* yuyv, unsigned char* rgb) {
        for (int i = 0; i < m_width * m_height / 2; i++) {
            int y0 = yuyv[i * 4 + 0];
            int u  = yuyv[i * 4 + 1];
            int y1 = yuyv[i * 4 + 2];
            int v  = yuyv[i * 4 + 3];

            auto clamp = [](int val) -> unsigned char {
                return static_cast<unsigned char>(val < 0 ? 0 : (val > 255 ? 255 : val));
            };

            int c = y0 - 16, d = u - 128, e = v - 128;
            rgb[i * 6 + 0] = clamp((298 * c + 409 * e + 128) >> 8);
            rgb[i * 6 + 1] = clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
            rgb[i * 6 + 2] = clamp((298 * c + 516 * d + 128) >> 8);

            c = y1 - 16;
            rgb[i * 6 + 3] = clamp((298 * c + 409 * e + 128) >> 8);
            rgb[i * 6 + 4] = clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
            rgb[i * 6 + 5] = clamp((298 * c + 516 * d + 128) >> 8);
        }
    }

    // MJPEG → RGB変換（stb_imageを使用）
    // 注意: この機能を使う場合は stb_image.h のインクルードが必要
    bool convertMJPEGtoRGB(const unsigned char* mjpeg, size_t len, unsigned char* rgb) {
// stb_imageがある場合
#ifdef STB_IMAGE_IMPLEMENTATION
        int w, h, ch;
        unsigned char* data = stbi_load_from_memory(mjpeg, (int)len, &w, &h, &ch, 3);
        if (data) {
            memcpy(rgb, data, w * h * 3);
            stbi_image_free(data);
            return true;
        }
#endif
        // stb_imageがない場合は黒画像
        memset(rgb, 0, m_width * m_height * 3);
        return false;
    }

public:
    SimpleCamera() = default;
    ~SimpleCamera() { close(); }

    // コピー禁止
    SimpleCamera(const SimpleCamera&) = delete;
    SimpleCamera& operator=(const SimpleCamera&) = delete;

    static std::vector<std::string> listDevices() {
        std::vector<std::string> devices;
        DIR* dir = opendir("/dev");
        if (dir) {
            struct dirent* entry;
            while ((entry = readdir(dir)) != nullptr) {
                std::string name = entry->d_name;
                if (name.find("video") == 0) {
                    devices.push_back("/dev/" + name);
                }
            }
            closedir(dir);
        }
        std::sort(devices.begin(), devices.end());
        return devices;
    }

    bool open(int deviceIndex = 0, int width = 640, int height = 480) {
        if (m_isOpened) close();

        std::string device = "/dev/video" + std::to_string(deviceIndex);
        m_fd = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (m_fd < 0) {
            std::cerr << "Cannot open: " << device << std::endl;
            return false;
        }

        // カメラの機能を確認
        v4l2_capability cap{};
        if (xioctl(m_fd, VIDIOC_QUERYCAP, &cap) < 0) {
            std::cerr << "Failed to query capabilities" << std::endl;
            close();
            return false;
        }

        // フォーマット設定（まずMJPEGを試す、失敗したらYUYV）
        v4l2_format fmt{};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (xioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0) {
            // MJPEGが失敗したらYUYVを試す
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            if (xioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0) {
                std::cerr << "Failed to set format" << std::endl;
                close();
                return false;
            }
        }

        m_width = fmt.fmt.pix.width;
        m_height = fmt.fmt.pix.height;
        m_pixelFormat = fmt.fmt.pix.pixelformat;

        std::cout << "Camera format: " << m_width << "x" << m_height;
        if (m_pixelFormat == V4L2_PIX_FMT_MJPEG) {
            std::cout << " (MJPEG)" << std::endl;
        } else if (m_pixelFormat == V4L2_PIX_FMT_YUYV) {
            std::cout << " (YUYV)" << std::endl;
        } else {
            std::cout << " (unknown format)" << std::endl;
        }

        // バッファ要求
        v4l2_requestbuffers req{};
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;

        if (xioctl(m_fd, VIDIOC_REQBUFS, &req) < 0) {
            std::cerr << "Failed to request buffers" << std::endl;
            close();
            return false;
        }

        m_buffers.resize(req.count);

        for (unsigned int i = 0; i < req.count; i++) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;

            if (xioctl(m_fd, VIDIOC_QUERYBUF, &buf) < 0) {
                close();
                return false;
            }

            m_buffers[i].length = buf.length;
            m_buffers[i].start = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                                      MAP_SHARED, m_fd, buf.m.offset);

            if (m_buffers[i].start == MAP_FAILED) {
                close();
                return false;
            }
        }

        // バッファをキューに入れる
        for (unsigned int i = 0; i < m_buffers.size(); i++) {
            v4l2_buffer buf{};
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            xioctl(m_fd, VIDIOC_QBUF, &buf);
        }

        // ストリーム開始
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(m_fd, VIDIOC_STREAMON, &type) < 0) {
            close();
            return false;
        }

        m_rgbBuffer.resize(m_width * m_height * 3);
        m_isOpened = true;

        std::cout << "Camera opened: " << m_width << "x" << m_height << std::endl;
        return true;
    }

    bool captureFrame(unsigned char* rgbOut) {
        if (!m_isOpened) return false;

        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(m_fd, &fds);

        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout

        int r = select(m_fd + 1, &fds, nullptr, nullptr, &tv);
        if (r <= 0) return false;

        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (xioctl(m_fd, VIDIOC_DQBUF, &buf) < 0) {
            return false;
        }

        // フォーマットに応じて変換
        if (m_pixelFormat == V4L2_PIX_FMT_YUYV) {
            convertYUYVtoRGB(static_cast<unsigned char*>(m_buffers[buf.index].start), rgbOut);
        } else if (m_pixelFormat == V4L2_PIX_FMT_MJPEG) {
            convertMJPEGtoRGB(static_cast<unsigned char*>(m_buffers[buf.index].start),
                              buf.bytesused, rgbOut);
        }

        xioctl(m_fd, VIDIOC_QBUF, &buf);
        return true;
    }

    bool captureFrame(std::vector<unsigned char>& rgb) {
        rgb.resize(m_width * m_height * 3);
        return captureFrame(rgb.data());
    }

    void close() {
        if (m_fd >= 0) {
            v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            xioctl(m_fd, VIDIOC_STREAMOFF, &type);

            for (auto& buf : m_buffers) {
                if (buf.start && buf.start != MAP_FAILED) {
                    munmap(buf.start, buf.length);
                }
            }
            m_buffers.clear();

            ::close(m_fd);
            m_fd = -1;
        }
        m_isOpened = false;
    }

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    bool isOpened() const { return m_isOpened; }
};

#endif // SIMPLE_CAMERA_LINUX

//=============================================================================
// Windows実装 (DirectShow)
//=============================================================================
#ifdef SIMPLE_CAMERA_WINDOWS

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dshow.h>

#pragma comment(lib, "strmiids.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")

// ISampleGrabberCB インターフェース
MIDL_INTERFACE("0579154A-2B53-4994-B0D0-E773148EFF85")
ISampleGrabberCB : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE SampleCB(double SampleTime, IMediaSample *pSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE BufferCB(double SampleTime, BYTE *pBuffer, long BufferLen) = 0;
};

// ISampleGrabber インターフェース
MIDL_INTERFACE("6B652FFF-11FE-4fce-92AD-0266B5D7C78F")
ISampleGrabber : public IUnknown {
public:
    virtual HRESULT STDMETHODCALLTYPE SetOneShot(BOOL OneShot) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetMediaType(const AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetConnectedMediaType(AM_MEDIA_TYPE *pType) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetBufferSamples(BOOL BufferThem) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentBuffer(long *pBufferSize, long *pBuffer) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetCurrentSample(IMediaSample **ppSample) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetCallback(ISampleGrabberCB *pCallback, long WhichMethodToCallback) = 0;
};

static const CLSID CLSID_SampleGrabber = {0xC1F400A0, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};
static const CLSID CLSID_NullRenderer = {0xC1F400A4, 0x3F08, 0x11d3, {0x9F, 0x0B, 0x00, 0x60, 0x08, 0x03, 0x9E, 0x37}};
static const IID IID_ISampleGrabber = {0x6B652FFF, 0x11FE, 0x4fce, {0x92, 0xAD, 0x02, 0x66, 0xB5, 0xD7, 0xC7, 0x8F}};

class SimpleCamera {
private:
    IGraphBuilder* m_graph = nullptr;
    ICaptureGraphBuilder2* m_capture = nullptr;
    IMediaControl* m_control = nullptr;
    IBaseFilter* m_cameraFilter = nullptr;
    IBaseFilter* m_grabberFilter = nullptr;
    ISampleGrabber* m_grabber = nullptr;

    int m_width = 0;
    int m_height = 0;
    bool m_isOpened = false;
    std::vector<unsigned char> m_buffer;

    static void freeMediaType(AM_MEDIA_TYPE& mt) {
        if (mt.cbFormat != 0) {
            CoTaskMemFree(mt.pbFormat);
            mt.cbFormat = 0;
            mt.pbFormat = nullptr;
        }
        if (mt.pUnk != nullptr) {
            mt.pUnk->Release();
            mt.pUnk = nullptr;
        }
    }

public:
    SimpleCamera() {
        CoInitialize(nullptr);
    }

    ~SimpleCamera() {
        close();
        CoUninitialize();
    }

    // コピー禁止
    SimpleCamera(const SimpleCamera&) = delete;
    SimpleCamera& operator=(const SimpleCamera&) = delete;

    static std::vector<std::string> listDevices() {
        std::vector<std::string> devices;
        CoInitialize(nullptr);

        ICreateDevEnum* devEnum = nullptr;
        HRESULT hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr,
                                      CLSCTX_INPROC_SERVER, IID_ICreateDevEnum,
                                      (void**)&devEnum);
        if (SUCCEEDED(hr)) {
            IEnumMoniker* enumMoniker = nullptr;
            hr = devEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMoniker, 0);
            if (hr == S_OK && enumMoniker) {
                IMoniker* moniker = nullptr;
                while (enumMoniker->Next(1, &moniker, nullptr) == S_OK) {
                    IPropertyBag* propBag = nullptr;
                    hr = moniker->BindToStorage(nullptr, nullptr, IID_IPropertyBag, (void**)&propBag);
                    if (SUCCEEDED(hr)) {
                        VARIANT var;
                        VariantInit(&var);
                        hr = propBag->Read(L"FriendlyName", &var, nullptr);
                        if (SUCCEEDED(hr)) {
                            char name[256];
                            WideCharToMultiByte(CP_UTF8, 0, var.bstrVal, -1, name, sizeof(name), nullptr, nullptr);
                            devices.push_back(name);
                            VariantClear(&var);
                        }
                        propBag->Release();
                    }
                    moniker->Release();
                }
                enumMoniker->Release();
            }
            devEnum->Release();
        }
        return devices;
    }

    bool open(int deviceIndex = 0, int width = 640, int height = 480) {
        if (m_isOpened) close();

        HRESULT hr;

        // フィルターグラフ作成
        hr = CoCreateInstance(CLSID_FilterGraph, nullptr, CLSCTX_INPROC_SERVER,
                              IID_IGraphBuilder, (void**)&m_graph);
        if (FAILED(hr)) return false;

        hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, nullptr, CLSCTX_INPROC_SERVER,
                              IID_ICaptureGraphBuilder2, (void**)&m_capture);
        if (FAILED(hr)) return false;

        m_capture->SetFiltergraph(m_graph);

        // カメラデバイス取得
        ICreateDevEnum* devEnum = nullptr;
        hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                              IID_ICreateDevEnum, (void**)&devEnum);
        if (FAILED(hr)) return false;

        IEnumMoniker* enumMoniker = nullptr;
        hr = devEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMoniker, 0);
        if (hr != S_OK || !enumMoniker) {
            devEnum->Release();
            return false;
        }

        IMoniker* moniker = nullptr;
        int index = 0;
        while (enumMoniker->Next(1, &moniker, nullptr) == S_OK) {
            if (index == deviceIndex) {
                hr = moniker->BindToObject(nullptr, nullptr, IID_IBaseFilter, (void**)&m_cameraFilter);
                moniker->Release();
                break;
            }
            moniker->Release();
            index++;
        }
        enumMoniker->Release();
        devEnum->Release();

        if (!m_cameraFilter) return false;

        m_graph->AddFilter(m_cameraFilter, L"Camera");

        // SampleGrabber作成
        hr = CoCreateInstance(CLSID_SampleGrabber, nullptr, CLSCTX_INPROC_SERVER,
                              IID_IBaseFilter, (void**)&m_grabberFilter);
        if (FAILED(hr)) return false;

        m_grabberFilter->QueryInterface(IID_ISampleGrabber, (void**)&m_grabber);
        m_graph->AddFilter(m_grabberFilter, L"Grabber");

        // メディアタイプ設定（RGB24）
        AM_MEDIA_TYPE mt;
        ZeroMemory(&mt, sizeof(mt));
        mt.majortype = MEDIATYPE_Video;
        mt.subtype = MEDIASUBTYPE_RGB24;
        m_grabber->SetMediaType(&mt);

        // NullRenderer作成
        IBaseFilter* nullRenderer = nullptr;
        hr = CoCreateInstance(CLSID_NullRenderer, nullptr, CLSCTX_INPROC_SERVER,
                              IID_IBaseFilter, (void**)&nullRenderer);
        if (FAILED(hr)) return false;

        m_graph->AddFilter(nullRenderer, L"Null");

        // グラフを接続
        hr = m_capture->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video,
                                     m_cameraFilter, m_grabberFilter, nullRenderer);
        nullRenderer->Release();

        if (FAILED(hr)) return false;

        // 実際のサイズを取得
        AM_MEDIA_TYPE connectedMt;
        m_grabber->GetConnectedMediaType(&connectedMt);
        VIDEOINFOHEADER* vih = (VIDEOINFOHEADER*)connectedMt.pbFormat;
        m_width = vih->bmiHeader.biWidth;
        m_height = abs(vih->bmiHeader.biHeight);
        freeMediaType(connectedMt);

        m_grabber->SetBufferSamples(TRUE);
        m_grabber->SetOneShot(FALSE);

        // 開始
        m_graph->QueryInterface(IID_IMediaControl, (void**)&m_control);
        m_control->Run();

        m_buffer.resize(m_width * m_height * 3);
        m_isOpened = true;

        std::cout << "Camera opened: " << m_width << "x" << m_height << std::endl;
        return true;
    }

    bool captureFrame(unsigned char* rgbOut) {
        if (!m_isOpened || !m_grabber) return false;

        long size = m_buffer.size();
        HRESULT hr = m_grabber->GetCurrentBuffer(&size, (long*)m_buffer.data());

        if (SUCCEEDED(hr) && size > 0) {
            // BGRからRGBに変換 + 上下反転
            for (int y = 0; y < m_height; y++) {
                for (int x = 0; x < m_width; x++) {
                    int srcIdx = ((m_height - 1 - y) * m_width + x) * 3;
                    int dstIdx = (y * m_width + x) * 3;
                    rgbOut[dstIdx + 0] = m_buffer[srcIdx + 2];  // R
                    rgbOut[dstIdx + 1] = m_buffer[srcIdx + 1];  // G
                    rgbOut[dstIdx + 2] = m_buffer[srcIdx + 0];  // B
                }
            }
            return true;
        }
        return false;
    }

    bool captureFrame(std::vector<unsigned char>& rgb) {
        rgb.resize(m_width * m_height * 3);
        return captureFrame(rgb.data());
    }

    void close() {
        if (m_control) { m_control->Stop(); m_control->Release(); m_control = nullptr; }
        if (m_grabber) { m_grabber->Release(); m_grabber = nullptr; }
        if (m_grabberFilter) { m_grabberFilter->Release(); m_grabberFilter = nullptr; }
        if (m_cameraFilter) { m_cameraFilter->Release(); m_cameraFilter = nullptr; }
        if (m_capture) { m_capture->Release(); m_capture = nullptr; }
        if (m_graph) { m_graph->Release(); m_graph = nullptr; }
        m_isOpened = false;
    }

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    bool isOpened() const { return m_isOpened; }
};

#endif // SIMPLE_CAMERA_WINDOWS

//=============================================================================
// macOS実装（スタブ）
//=============================================================================
#ifdef SIMPLE_CAMERA_MACOS

class SimpleCamera {
public:
    SimpleCamera() {}
    ~SimpleCamera() {}

    static std::vector<std::string> listDevices() { return {}; }
    bool open(int deviceIndex = 0, int width = 640, int height = 480) {
        std::cerr << "macOS is not supported yet" << std::endl;
        return false;
    }
    bool captureFrame(unsigned char* rgbOut) { return false; }
    bool captureFrame(std::vector<unsigned char>& rgb) { return false; }
    void close() {}
    int getWidth() const { return 0; }
    int getHeight() const { return 0; }
    bool isOpened() const { return false; }
};

#endif // SIMPLE_CAMERA_MACOS
