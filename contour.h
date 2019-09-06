#ifndef CONTOUR_H
#define CONTOUR_H
#include <iostream>
#include <string>
#include <vector>
#include <QFile>
#include <QJsonParseError>
#include <QJsonObject>
#include <QJsonArray>
/**
 * @brief Defines which 2D landmarks comprise the right and left face contour.
 *
 * This class holds 2D image contour landmark information. More specifically,
 * it defines which 2D landmark IDs correspond to the right contour and which
 * to the left. These definitions are loaded from a file, for example from
 * the "contour_landmarks" part of share/ibug_to_sfm.txt.
 *
 * Todo: We should improve error handling here. When there's no contour_landmarks
 * in the file, it will crash, but it would be nice if it still worked, the returned
 * vectors should just be empty.
 *
 * Note: Better names could be ContourDefinition or ImageContourLandmarks, to
 * disambiguate 3D and 2D landmarks?
 * Todo: I think this should go into the LandmarkMapper. Isn't it part of ibug_to_sfm.txt already?
 */
struct ContourLandmarks
{
    // starting from right side, eyebrow-height.
    std::vector<long long> rightContour;
    // Chin point is not included in the contour here.
    // starting from left side, eyebrow-height. Order doesn't matter here.
    std::vector<long long> leftContour;

    // Note: We store r/l separately because we currently only fit to the contour facing the camera.

    /**
     * Helper method to load contour landmarks from a text file with landmark
     * mappings, like ibug_to_sfm.txt.
     *
     * @param[in] filename Filename to a landmark-mapping file.
     * @return A ContourLandmarks instance with loaded 2D contour landmarks.
     * @throw std::runtime_error runtime_error or toml::exception if there is an error loading the landmarks
     * from the file.
     */
    static ContourLandmarks load(std::string jsonFile)
    {
        ContourLandmarks contour;
        // There might be a vector<string> or a vector<int>, we need to check for that and convert to vector<string>.
        // Todo: Again, check whether the key exists first.
        // Read all the "right" contour landmarks:
        QFile loadFile(QString::fromStdString(jsonFile));
        if(!loadFile.open(QIODevice::ReadOnly))
        {
            std::cout << "could't open projects json";
            exit(EXIT_FAILURE);
        }
        QByteArray allData = loadFile.readAll();
        loadFile.close();

        QJsonParseError json_error;
        QJsonDocument jsonDoc(QJsonDocument::fromJson(allData, &json_error));

        if(json_error.error != QJsonParseError::NoError)
        {
            std::cout << "json error!";
            exit(EXIT_FAILURE);
        }
        QJsonObject obj=jsonDoc.object();
        QJsonObject contourObj=obj.value("contour_landmarks").toObject();
        QJsonArray rightContour = contourObj.value("right").toArray();
        for (int j = 0;j<rightContour.size();j++)
        {
            int value=rightContour[j].toInt()-1;
            contour.rightContour.push_back(value);
        }
        // Now the same for all the "left" contour landmarks:
        QJsonArray leftContour = contourObj.value("left").toArray();
        for (int j = 0;j<leftContour.size();j++)
        {
            int value=leftContour[j].toInt()-1;
            contour.leftContour.push_back(value);
        }
        return contour;
    };
};
#endif // CONTOUR_H
