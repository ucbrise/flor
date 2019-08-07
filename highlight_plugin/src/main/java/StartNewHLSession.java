import com.intellij.openapi.actionSystem.AnAction;
import com.intellij.openapi.actionSystem.AnActionEvent;
import com.intellij.openapi.actionSystem.CommonDataKeys;
import com.intellij.openapi.editor.Editor;
import com.intellij.openapi.project.Project;
import com.intellij.openapi.project.ProjectManager;
import com.intellij.openapi.vfs.VirtualFile;
import com.intellij.psi.PsiFile;
import org.jdom.JDOMException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.SystemIndependent;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.*;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;

public class StartNewHLSession extends AnAction {
    @Override
    public void actionPerformed(@NotNull AnActionEvent e) {
        Project project = e.getProject();
        Editor editor = e.getData(CommonDataKeys.EDITOR);
        PsiFile file = e.getData(CommonDataKeys.PSI_FILE);
        if (file == null) {
            return;
        }
        VirtualFile vFile = file.getVirtualFile();
        @SystemIndependent String projectBasePath = project.getBasePath();
        String currFolderName = vFile.getParent().getName();
        String timeStamp = new SimpleDateFormat("yyyyMMdd-HHmmss").format(new Date());
        String dstFolderPath = projectBasePath + "/" + "highlighted." + timeStamp + ".d";

        File dstFlorPluginFolderPath = new File(projectBasePath + "/.flor_highlight");
        if (!dstFlorPluginFolderPath.exists()) {
            dstFlorPluginFolderPath.mkdir();
        }

        String configFilePath = dstFlorPluginFolderPath + "/config.xml";
        File configFile = new File(configFilePath);

        DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder documentBuilder = null;
        try {
            documentBuilder = documentFactory.newDocumentBuilder();
        } catch (ParserConfigurationException ex) {
            ex.printStackTrace();
        }
        Document document = null;

        TransformerFactory transformerFactory = TransformerFactory.newInstance();
        transformerFactory.setAttribute("indent-number", 2);
        Transformer transformer = null;
        try {
            transformer = transformerFactory.newTransformer();
        } catch (TransformerConfigurationException ex) {
            ex.printStackTrace();
        }

        assert transformer != null : "transformer is null";
        transformer.setOutputProperty(OutputKeys.INDENT, "yes");

        Element root = null, session = null;
        int numSessions = 1;

        assert documentBuilder != null : "documentBuilder is null";

        if (configFile.exists()) {
            // Append new session to existing config file
            try {

                document = documentBuilder.parse(configFile);
                root = document.getDocumentElement();

                numSessions = Integer.parseInt(root.getAttribute("numSessions")) + 1;
                root.setAttribute("numSessions", String.valueOf(numSessions));

                // session element
                session = document.createElement("session");

            } catch (SAXException | IOException ex) {
                ex.printStackTrace();
            }


        } else {
            // Create new config file and save

            document = documentBuilder.newDocument();
            // root element
            root = document.createElement("project");
            document.appendChild(root);
            root.setAttribute("numSessions", String.valueOf(1));

            // session element
            session = document.createElement("session");

            System.out.println("Done creating XML Config File");

        }
        // set session's id attribute
        session.setAttribute("id", String.valueOf(numSessions));
        session.setIdAttribute("id", true);


        // add timestamp element
        assert document != null : "document is null";
        Element tsElement = document.createElement("timestamp");
        tsElement.appendChild(document.createTextNode(timeStamp));
        session.appendChild(tsElement);

        // add path element to save new working directory in config file
        Element pathElement = document.createElement("path");
        File dstDir = new File(dstFolderPath);
        dstDir.mkdir();
        pathElement.appendChild(document.createTextNode(dstFolderPath));
        session.appendChild(pathElement);

        root.appendChild(session);

        // create the xml file
        // transform the DOM Object to an XML File
        DOMSource domSource = new DOMSource(document);
        StreamResult streamResult = new StreamResult(configFile);

        // Use
        // StreamResult result = new StreamResult(System.out);
        // for debugging

        try {
            transformer.transform(domSource, streamResult);
            System.out.println("Done updating config file with new session config.");
        } catch (TransformerException ex) {
            ex.printStackTrace();
        }


    }
}
