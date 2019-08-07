import com.intellij.openapi.actionSystem.AnActionEvent;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import java.io.File;
import java.io.IOException;

public class SessionManager {

    private File dstFlorPluginFolder;

    public SessionManager(String projectDir) {
        dstFlorPluginFolder = new File(projectDir + "/.flor_highlight");
    }

    public String getLatestSessionFolderPath(AnActionEvent e) {
        
        String path = null;

        String configFilePath = dstFlorPluginFolder + "/config.xml";
        File configFile = new File(configFilePath);

        if (!configFile.exists()) {
            new StartNewHLSession().actionPerformed(e);
            System.out.println("No session found! Auto starting a new session...");
        }

        DocumentBuilderFactory documentFactory = DocumentBuilderFactory.newInstance();
        DocumentBuilder documentBuilder = null;
        try {
            documentBuilder = documentFactory.newDocumentBuilder();
        } catch (ParserConfigurationException ex) {
            ex.printStackTrace();
        }

        int numSessions = 0;

        try {
            Document document = null;
            if (documentBuilder != null) {
                document = documentBuilder.parse(configFile);
            }
            Element root = null;
            if (document != null) {
                root = document.getDocumentElement();
            }
            if (root != null) {
                numSessions = Integer.parseInt(root.getAttribute("numSessions"));
            }
            Element session = null;
            if (root != null) {
                NodeList sessionList = document.getElementsByTagName("session");
                for (int i = 0; i < sessionList.getLength(); i++) {
                    Element s = (Element) sessionList.item(i);
                    s.setIdAttribute("id", true);
                }
                session = document.getElementById(String.valueOf(numSessions));
            }

            if (session != null) {
                path = session.getElementsByTagName("path").item(0).getFirstChild().getTextContent();
            }
        } catch (SAXException | IOException ex) {
            ex.printStackTrace();
        }
        
        return path;
    }
}
