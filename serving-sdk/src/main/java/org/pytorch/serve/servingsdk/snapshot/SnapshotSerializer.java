package org.pytorch.serve.servingsdk.snapshot;


import java.io.IOException;

public interface SnapshotSerializer {

    void saveSnapshot(Snapshot snapshot) throws IOException;

    Snapshot getSnapshot(String modelSnapshot) throws IOException;
}
